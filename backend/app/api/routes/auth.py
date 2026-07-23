import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import UserRegister, UserResponse, Token
from app.core.security import get_password_hash, verify_password, create_access_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


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
    if not user or not verify_password(user_in.password, user.hashed_password):
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
    if not user or not verify_password(form_data.password, user.hashed_password):
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
