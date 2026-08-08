from functools import wraps
from flask import request

def validate_json(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return {'error': 'JSON required'}, 400
        return f(*args, **kwargs)
    return decorated

def require_fields(*fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            data = request.json
            for field in fields:
                if field not in data:
                    return {'error': f'Missing {field}'}, 400
            return f(*args, **kwargs)
        return decorated
    return decorator
