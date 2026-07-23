from typing import Optional
from pydantic import BaseModel, Field

class UserRegister(BaseModel):
    email: str = Field(..., min_length=3, max_length=100, description="User's email address")
    password: str = Field(..., min_length=6, max_length=100, description="Password (min 6 characters)")

class UserResponse(BaseModel):
    user_id: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: Optional[str] = None
