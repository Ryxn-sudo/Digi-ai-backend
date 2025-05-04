from src.server import create_app

if __name__ == '__main__':
    app = create_app()
    print("Starting Handwritten Digit Recognition Server...")
    print("Access the server at http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
