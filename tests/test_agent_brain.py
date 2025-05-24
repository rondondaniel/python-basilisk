"""
Test script to verify AgentBrain class functionality
"""
import logging
# Fix path for imports
from tests.context import *
from agent_brain import AgentBrain
from models import EmotionTypes, ActionTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent_brain():
    """Test the AgentBrain class functionality"""
    try:
        logger.info("Initializing AgentBrain...")
        brain = AgentBrain()
        
        # Test emotion tool - get current emotion
        logger.info("Testing emotion_tool - get current emotion...")
        emotion_response = brain._emotion_tool()
        logger.info(f"Response: {emotion_response}")
        
        # Test emotion tool - set emotion
        logger.info("Testing emotion_tool - set emotion to HUNGRY...")
        emotion_set_response = brain._emotion_tool('{"emotion": "HUNGRY"}')
        logger.info(f"Response: {emotion_set_response}")
        
        # Test emotion tool - get updated emotion
        logger.info("Testing emotion_tool - get updated emotion...")
        updated_emotion = brain._emotion_tool()
        logger.info(f"Response: {updated_emotion}")
        
        # Test move tool - move right
        logger.info("Testing move_tool - move RIGHT...")
        move_response = brain._move_tool('{"direction": "RIGHT"}')
        logger.info(f"Response: {move_response}")
        
        # Test with invalid inputs
        logger.info("Testing with invalid emotion...")
        invalid_emotion = brain._emotion_tool('{"emotion": "HAPPY"}')
        logger.info(f"Response: {invalid_emotion}")
        
        logger.info("Testing with invalid direction...")
        invalid_direction = brain._move_tool('{"direction": "NORTHEAST"}')
        logger.info(f"Response: {invalid_direction}")
        
        logger.info("✅ All tests completed.")
        return True
    except Exception as e:
        logger.error(f"❌ Error testing AgentBrain: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting AgentBrain test...")
    success = test_agent_brain()
    if success:
        logger.info("✅ Test completed successfully.")
    else:
        logger.error("❌ Test failed. Please check the error messages above.")
