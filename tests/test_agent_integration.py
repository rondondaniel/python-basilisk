"""
Test script to verify the complete integration of Agent, AgentBrain, and BoardEnv
"""
import logging
# Fix path for imports
from tests.context import *
from agent import Agent
from env import BoardEnv
from models import EmotionTypes, ActionTypes, CellTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent_integration():
    """Test the complete integration of Agent, AgentBrain and BoardEnv"""
    try:
        # Create environment
        logger.info("Creating environment...")
        env = BoardEnv()
        
        # Create agent separately from environment
        logger.info("Creating agent (without environment)...")
        agent = Agent()
        
        # Initialize agent state from environment
        logger.info("Initializing agent state from environment...")
        agent.initialize_state_from_env(env)
        
        # Test observation with explicit environment passing
        logger.info("Testing agent observation with explicit environment passing...")
        observation = agent.observe(env)
        logger.info(f"Observation: {observation}")
        
        # Test action with explicit environment passing
        logger.info("Testing agent movement with explicit environment passing...")
        action_result = agent.take_action("right", env)
        logger.info(f"Action result: {action_result}")
        
        # Test emotion setting
        logger.info("Testing emotion setting...")
        emotion_result = agent.set_emotion("HUNGRY")
        logger.info(f"Emotion result: {emotion_result}")
        
        # Test processing a query with the LLM
        logger.info("Testing LLM processing with environment context...")
        query = "Where am I and what's around me?"
        llm_result = agent.process_query(query, env)
        logger.info(f"Query result: {llm_result}")
        
        logger.info("✅ All integration tests completed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error in integration test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting agent integration test...")
    success = test_agent_integration()
    if success:
        logger.info("✅ Integration test completed successfully.")
    else:
        logger.error("❌ Integration test failed. Please check the error messages above.")
