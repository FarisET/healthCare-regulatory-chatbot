from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# Import your chatbot
try:
    from chatbot6 import rag_agent
    print("✓ Successfully imported rag_agent")
except Exception as e:
    print(f"✗ Failed to import rag_agent: {e}")
    rag_agent = None

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Add CORS headers to every response
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Handle preflight request
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        # Get the message
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data:
            return jsonify({"reply": "Error: No data received"}), 400
        
        message = data.get("message", "")
        print(f"Message: {message}")
        
        if not message:
            return jsonify({"reply": "Error: Empty message"}), 400
        
        # Check if agent is available
        if rag_agent is None:
            return jsonify({"reply": "Error: Chatbot agent not initialized"}), 500
        
        # Call the agent
        print(f"Calling agent with message: {message}")
        result = rag_agent.invoke({"messages": [{"role": "user", "content": message}]})
        
        print(f"Agent result type: {type(result)}")
        print(f"Agent result: {result}")
        
        # ✅ FIX: Extract reply from AIMessage object (not dict)
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]
            
            # Check if it's a LangChain message object (has .content attribute)
            if hasattr(last_message, "content"):
                reply = last_message.content
            # Or if it's a dict with 'content' key
            elif isinstance(last_message, dict) and "content" in last_message:
                reply = last_message["content"]
            else:
                reply = str(last_message)
        else:
            reply = str(result)
        
        print(f"Reply: {reply}")
        
        return jsonify({"reply": reply})
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Exception occurred: {error_msg}")
        traceback.print_exc()
        return jsonify({"reply": error_msg}), 500

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == "__main__":
    print("Starting Flask server on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)