import logging
import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import httpx

from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserResponse, Token, GoogleAuthRequest
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# ── Google OAuth helper ──────────────────────────────────

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


async def _exchange_google_code(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for tokens, then fetch user info."""
    async with httpx.AsyncClient() as client:
        # Step 1: Exchange code for access token
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

        # Step 2: Fetch user info using the access token
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


# ── Endpoints ────────────────────────────────────────────


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserRegister, db: Session = Depends(get_db)) -> UserResponse:
    existing_user = db.query(User).filter(User.email == user_in.email).first()
    if existing_user:
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

    # If user signed up via Google, block email/password login
    if user and user.auth_provider == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In. Please sign in with Google."
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
    
    access_token = create_access_token(subject=user.user_id)
    return Token(access_token=access_token)


@router.post("/token", response_model=Token)
async def login_oauth2(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == form_data.username).first()

    if user and user.auth_provider == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In. Please sign in with Google."
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
    
    access_token = create_access_token(subject=user.user_id)
    return Token(access_token=access_token)


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

    # Exchange the auth code for user info
    google_user = await _exchange_google_code(body.code, body.redirect_uri)
    google_id = google_user.get("sub")
    email = google_user.get("email")

    if not google_id or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve email from Google account.",
        )

    # Look up user by Google ID first, then by email
    user = db.query(User).filter(User.google_id == google_id).first()
    if not user:
        user = db.query(User).filter(User.email == email).first()

    if user:
        # Link Google ID if not already set (auto-link existing email/password accounts)
        if not user.google_id:
            user.google_id = google_id
            user.auth_provider = "google"
            db.commit()
            logger.info("Linked Google account to existing user email=%s", email)
    else:
        # Create new user
        user = User(
            email=email,
            hashed_password=None,
            auth_provider="google",
            google_id=google_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created new Google user email=%s (id=%s)", email, user.user_id)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is deactivated.",
        )

    access_token = create_access_token(subject=user.user_id)
    return Token(access_token=access_token)
