import base64
from unittest.mock import patch

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import Session

from app.image_agent import ImageAgent


@pytest.mark.asyncio
async def test_image_agent_success():
    # Create a mock InvocationContext
    ctx = InvocationContext(
        session=Session(
            state={
                "current_recipe": {
                    "title": "Test Recipe",
                    "description": "A delicious test recipe.",
                }
            }
        )
    )

    # Mock the generate_recipe_image function
    with patch(
        "app.image_agent.generate_recipe_image",
        return_value=base64.b64encode(b"test_image_data").decode("utf-8"),
    ) as mock_generate_image:
        # Create and run the ImageAgent
        agent = ImageAgent(name="test_image_agent")
        events = [event async for event in agent.run(ctx)]

        # Assert that the generate_recipe_image function was called with the correct arguments
        mock_generate_image.assert_called_once_with(
            "Test Recipe", "A delicious test recipe."
        )

        # Assert that the agent produced the expected event
        assert len(events) == 1
        assert events[0].author == "test_image_agent"
        assert events[0].actions.state_delta["final_recipe_image"] == base64.b64encode(
            b"test_image_data"
        ).decode("utf-8")


@pytest.mark.asyncio
async def test_image_agent_failure():
    # Create a mock InvocationContext
    ctx = InvocationContext(
        session=Session(
            state={
                "current_recipe": {
                    "title": "Test Recipe",
                    "description": "A delicious test recipe.",
                }
            }
        )
    )

    # Mock the generate_recipe_image function to return None
    with patch(
        "app.image_agent.generate_recipe_image", return_value=None
    ) as mock_generate_image:
        # Create and run the ImageAgent
        agent = ImageAgent(name="test_image_agent")
        events = [event async for event in agent.run(ctx)]

        # Assert that the generate_recipe_image function was called with the correct arguments
        mock_generate_image.assert_called_once_with(
            "Test Recipe", "A delicious test recipe."
        )

        # Assert that the agent produced the expected event
        assert len(events) == 1
        assert events[0].author == "test_image_agent"
        assert events[0].actions.state_delta["final_recipe_image"] is None
