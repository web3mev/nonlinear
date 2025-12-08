print("Starting import...")
try:
    import dash_app
    print("Import Successful")
except Exception as e:
    print(f"Import Failed: {e}")
    import traceback
    traceback.print_exc()
