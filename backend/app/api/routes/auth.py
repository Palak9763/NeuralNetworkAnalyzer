import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import httpx

from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserResponse, Token, GoogleAuthRequest, GitHubAuthRequest
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

OAUTH_PROVIDER_LABELS = {
    "google": "Google",
    "github": "GitHub",
}

# ── Google OAuth helper ──────────────────────────────────

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# ── GitHub OAuth helper ──────────────────────────────────

GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def _oauth_login_message(provider: str) -> str:
    label = OAUTH_PROVIDER_LABELS.get(provider, provider.title())
    return f"This account uses {label} Sign-In. Please sign in with {label}."


def _token_for_user(user: User) -> Token:
    return Token(
        access_token=create_access_token(subject=user.user_id),
        email=user.email,
    )


async def _exchange_google_code(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for tokens, then fetch user info."""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })
        if token_resp.status_code != 200:
            logger.error("Google token exchange failed: %s", token_resp.text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange Google authorization code.",
            )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")

        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            logger.error("Google userinfo fetch failed: %s", userinfo_resp.text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to fetch Google user info.",
            )
        return userinfo_resp.json()


async def _exchange_github_code(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for GitHub access token and fetch profile."""
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        if token_resp.status_code != 200:
            logger.error("GitHub token exchange failed: %s", token_resp.text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to exchange GitHub authorization code.",
            )

        token_data = token_resp.json()
        if token_data.get("error"):
            logger.error("GitHub token exchange error: %s", token_data)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=token_data.get("error_description") or "GitHub authorization failed.",
            )

        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GitHub did not return an access token.",
            )

        auth_headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        user_resp = await client.get(GITHUB_USER_URL, headers=auth_headers)
        if user_resp.status_code != 200:
            logger.error("GitHub user fetch failed: %s", user_resp.text)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to fetch GitHub user info.",
            )
        github_user = user_resp.json()

        email = github_user.get("email")
        if not email:
            emails_resp = await client.get(GITHUB_EMAILS_URL, headers=auth_headers)
            if emails_resp.status_code != 200:
                logger.error("GitHub emails fetch failed: %s", emails_resp.text)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to fetch GitHub account email.",
                )
            emails = emails_resp.json()
            primary = next((item for item in emails if item.get("primary")), None)
            verified = next((item for item in emails if item.get("verified")), None)
            chosen = primary or verified or (emails[0] if emails else None)
            email = chosen.get("email") if chosen else None

        github_user["resolved_email"] = email
        return github_user


def _find_or_create_oauth_user(
    db: Session,
    *,
    provider: str,
    provider_user_id: str,
    email: str,
    id_field: str,
) -> User:
    user = db.query(User).filter(getattr(User, id_field) == provider_user_id).first()
    if not user:
        user = db.query(User).filter(User.email == email).first()

    if user:
        if not getattr(user, id_field):
            setattr(user, id_field, provider_user_id)
            user.auth_provider = provider
            db.commit()
            logger.info("Linked %s account to existing user email=%s", provider, email)
        return user

    user = User(
        email=email,
        hashed_password=None,
        auth_provider=provider,
        **{id_field: provider_user_id},
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("Created new %s user email=%s (id=%s)", provider, email, user.user_id)
    return user


# ── Endpoints ────────────────────────────────────────────


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserRegister, db: Session = Depends(get_db)) -> UserResponse:
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
        if existing_user.auth_provider != "local":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_oauth_login_message(existing_user.auth_provider),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists."
        )

    try:
        new_user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            auth_provider="local",
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info("Registered new user email=%s (id=%s)", new_user.email, new_user.user_id)
        return new_user
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to register user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not register user."
        ) from exc


@router.post("/login", response_model=Token)
async def login_user(user_in: UserRegister, db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == user_in.email).first()

    if user and user.auth_provider != "local":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_oauth_login_message(user.auth_provider),
        )

    if not user or not verify_password(user_in.password, user.hashed_password or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated."
        )

    return _token_for_user(user)


@router.post("/token", response_model=Token)
async def login_oauth2(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == form_data.username).first()

    if user and user.auth_provider != "local":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_oauth_login_message(user.auth_provider),
        )

    if not user or not verify_password(form_data.password, user.hashed_password or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated."
        )

    return _token_for_user(user)


@router.post("/google", response_model=Token)
async def google_login(body: GoogleAuthRequest, db: Session = Depends(get_db)) -> Token:
    """
    Sign in (or register) via Google OAuth.

    Flow:
    1. Frontend sends the authorization code obtained from the Google popup
    2. We exchange it for tokens and fetch the user's Google profile
    3. Find or create the user in our database
    4. Return our own JWT token
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google OAuth is not configured on the server.",
        )

    google_user = await _exchange_google_code(body.code, body.redirect_uri)
    google_id = google_user.get("sub")
    email = google_user.get("email")

    if not google_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve email from Google account.",
        )

    user = _find_or_create_oauth_user(
        db,
        provider="google",
        provider_user_id=google_id,
        email=email,
        id_field="google_id",
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated.",
        )

    return _token_for_user(user)


@router.post("/github", response_model=Token)
async def github_login(body: GitHubAuthRequest, db: Session = Depends(get_db)) -> Token:
    """Sign in (or register) via GitHub OAuth."""
    if not settings.github_client_id or not settings.github_client_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub OAuth is not configured on the server.",
        )

    github_user = await _exchange_github_code(body.code, body.redirect_uri)
    github_id = str(github_user.get("id")) if github_user.get("id") is not None else None
    email = github_user.get("resolved_email")

    if not github_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve email from GitHub account. Ensure your GitHub email is verified.",
        )

    user = _find_or_create_oauth_user(
        db,
        provider="github",
        provider_user_id=github_id,
        email=email,
        id_field="github_id",
    )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated.",
        )

    return _token_for_user(user)
