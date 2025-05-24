"""
Test script to verify Ollama integration with LangChain
"""
import logging
from langchain_ollama import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """Test basic connection to Ollama server and model availability"""
    try:
        logger.info("Initializing Ollama LLM...")
        # Initialize the OllamaLLM with the same parameters as in your AgentBrain
        llm = OllamaLLM(
            model="llama3.2:1b",  # 1.2B parameter version
            base_url="http://127.0.0.1:11434",  # Ollama server URL
            temperature=0.7
        )
        
        # Simple prompt to test the connection
        logger.info("Testing LLM with a simple prompt...")
        response = llm.invoke("Hello, are you working correctly?")
        
        logger.info(f"✅ Success! Ollama responded with: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing Ollama connection: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Ollama integration test...")
    success = test_ollama_connection()
    if success:
        logger.info("✅ Test completed successfully. Ollama integration is working.")
    else:
        logger.error("❌ Test failed. Please check the error messages above.")
