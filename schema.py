from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int 
    os: str
    source: str 
    
class PostGet(BaseModel):
    id: int
    text: str
    topic: Optional[str] = None
    
class FeedGet(BaseModel):
    user_id: int
    post_id: int
    user: UserGet
    post: PostGet
    action: str
    time: datetime
