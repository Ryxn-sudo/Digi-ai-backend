import os
from src.server import create_app

# Create the Flask app
app = create_app()

# Run the server when script is executed
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Handwritten Digit Recognition Server on port {port}...")
    print(f"Access the server at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)
