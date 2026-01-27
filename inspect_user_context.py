try:
    from coreason_identity.models import UserContext
    print("UserContext imported successfully")
    print(UserContext.model_fields.keys())
except ImportError:
    print("Could not import UserContext")
except Exception as e:
    print(f"Error: {e}")
