# Tiny Tastes App Quickstart

The **Tiny Tastes App Quickstart** is a production-ready blueprint for building a sophisticated, fullstack research agent with Gemini. It's built to demonstrate how the ADK helps structure complex agentic workflows, build modular agents, and incorporate critical Human-in-the-Loop (HITL) steps.

<table>
  <thead>
    <tr>
      <th colspan="2">Key Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🏗️</td>
      <td><strong>Fullstack & Production-Ready:</strong> A complete React frontend and ADK-powered FastAPI backend, with deployment options for <a href="https://cloud.google.com/run">Google Cloud Run</a> and <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview">Vertex AI Agent Engine</a>.</td>
    </tr>
    <tr>
      <td>🧠</td>
      <td><strong>Advanced Agentic Workflow:</strong> The agent uses Gemini to <strong>strategize</strong> a multi-step plan, <strong>reflect</strong> on findings to identify gaps, and <strong>synthesize</strong> a final, comprehensive report.</td>
    </tr>
    <tr>
      <td>🔄</td>
      <td><strong>Iterative & Human-in-the-Loop Research:</strong> Involves the user for plan approval, then autonomously loops through searching (via Gemini function calling) and refining its results until it has gathered sufficient information.</td>
    </tr>
  </tbody>
</table>

Here is the agent in action:

<img src="https://github.com/GoogleCloudPlatform/agent-starter-pack/blob/main/docs/images/adk_gemini_fullstack.gif?raw=true" width="80%" alt="Tiny Tastes App Preview">

This project adapts concepts from the [Gemini FullStack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) for the frontend app.

This project is based on the original [Gemini Fullstack Agent Development Kit (ADK) Quickstart](https://github.com/google/adk-samples/tree/main/python/agents/gemini-fullstack).

## 🚀 Getting Started: From Zero to Running Agent in 1 Minute
**Prerequisites:** **[Python 3.10+](https://www.python.org/downloads/)**, **[Node.js](https://nodejs.org/)**, **[uv](https://github.com/astral-sh/uv)**

You have two options to get started. Choose the one that best fits your setup:

*   A. **[Google AI Studio](#a-google-ai-studio)**: Choose this path if you want to use a **Google AI Studio API key**. This method involves cloning the sample repository.
*   B. **[Google Cloud Vertex AI](#b-google-cloud-vertex-ai)**: Choose this path if you want to use an existing **Google Cloud project** for authentication. This method generates a new, prod-ready project using the [agent-starter-pack](https://goo.gle/agent-starter-pack) including all the deployment scripts required.

---

### A. Google AI Studio

You'll need a **[Google AI Studio API Key](https://aistudio.google.com/app/apikey)**.

#### Step 1: Clone Repository
Clone the repository and `cd` into the project directory.

```bash
git clone https://github.com/google/adk-samples.git
cd adk-samples/python/agents/gemini-fullstack
```

#### Step 2: Set Environment Variables
Create a `.env` file in the `app` folder by running the following command (replace YOUR_AI_STUDIO_API_KEY with your actual API key):

```bash
echo "GOOGLE_GENAI_USE_VERTEXAI=FALSE" >> app/.env
echo "GOOGLE_API_KEY=YOUR_AI_STUDIO_API_KEY" >> app/.env
```

#### Step 3: Install & Run
From the `gemini-fullstack` directory, install dependencies and start the servers.

```bash
make install && make dev
```
Your agent is now running at `http://localhost:5173`.

---

### B. Google Cloud Vertex AI

You'll also need: **[Google Cloud SDK](https://cloud.google.com/sdk/docs/install)** and a **Google Cloud Project** with the **Vertex AI API** enabled.

#### Step 1: Create Project from Template
This command uses the [Agent Starter Pack](https://goo.gle/agent-starter-pack) to create a new directory (`my-tiny-tastes-app`) with all the necessary code.
```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install the starter pack and create your project
pip install --upgrade agent-starter-pack
agent-starter-pack create my-tiny-tastes-app -a adk_gemini_fullstack
```
<details>
<summary>⚡️ Alternative: Using uv</summary>

If you have [`uv`](https://github.com/astral-sh/uv) installed, you can create and set up your project with a single command:
```bash
uvx agent-starter-pack create my-tiny-tastes-app -a adk_gemini_fullstack
```
This command handles creating the project without needing to pre-install the package into a virtual environment.
</details>

You'll be prompted to select a deployment option (Agent Engine or Cloud Run) and verify your Google Cloud credentials.

#### Step 2: Install & Run
Navigate into your **newly created project folder**, then install dependencies and start the servers.
```bash
cd my-tiny-tastes-app && make install && make dev
```
Your agent is now running at `http://localhost:5173`.

## ☁️ Cloud Deployment
> **Note:** The cloud deployment instructions below apply only if you chose the **Google Cloud Vertex AI** option.

You can quickly deploy your agent to a **development environment** on Google Cloud. You can deploy your latest code at any time with:

```bash
# Replace YOUR_DEV_PROJECT_ID with your actual Google Cloud Project ID
gcloud config set project YOUR_DEV_PROJECT_ID
make backend
```

For robust, **production-ready deployments** with automated CI/CD, please follow the detailed instructions in the **[Agent Starter Pack Development Guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/development-guide.html#b-production-ready-deployment-with-ci-cd)**.
## Agent Details

| Attribute | Description |
| :--- | :--- |
| **Interaction Type** | Workflow |
| **Complexity** | Advanced |
| **Agent Type** | Multi Agent |
| **Components** | Multi-agent, Function calling, Web search, React frontend, Human-in-the-Loop |
| **Vertical** | Horizontal |

## How the Agent Thinks: A Multi-Agent Workflow for Baby Food Recipes

The backend, defined in `app/agent.py`, uses a series of specialized agents to help users create safe and healthy baby food recipes.

The following diagram illustrates the agent's architecture and workflow:

![Tiny Tastes App Architecture](https://github.com/GoogleCloudPlatform/agent-starter-pack/blob/main/docs/images/adk_gemini_fullstack_architecture.png?raw=true)

The process involves several key agents:

1.  **`interactive_recipe_agent` (Root Agent):**
    *   **Description:** The primary assistant for creating baby food recipes. It collaborates with the user to generate a recipe and then gets it approved before finalization.
    *   **Workflow:**
        1.  **Generate Recipe:** When the user provides ingredients or a recipe name, it calls the `recipe_generator`.
        2.  **Present for Approval:** Shows the generated recipe to the user for their initial approval.
        3.  **Delegate to Pipeline:** If the user approves, it passes the recipe to the `recipe_creation_pipeline` for detailed review and refinement.

2.  **`recipe_generator`:**
    *   **Description:** Generates a creative and simple toddler food recipe (1-2 years old) based on user input (ingredients or recipe name).
    *   **Output:** A `Recipe` object containing the title, description, ingredients, instructions, and age appropriateness.

3.  **`recipe_creation_pipeline` (Sequential Agent):**
    *   **Description:** Manages the iterative refinement of the recipe.
    *   **Sub-agents:**
        *   **`iterative_refinement_loop` (Loop Agent):**
            *   **`pediatrician_critic_agent`:** Evaluates the current recipe for safety, nutritional value, and age-appropriateness. Outputs a `PediatricianFeedback` object with a "pass" or "fail" grade, comments, and follow-up questions if it fails.
            *   **`EscalationChecker`:** Checks the pediatrician's feedback. If the recipe "passes", it stops the loop.
            *   **`recipe_refiner_agent`:** If the recipe "fails", this agent uses the pediatrician's feedback and Google Search (for the follow-up questions) to revise and improve the recipe.
        *   **`final_recipe_presenter_agent`:** Once the recipe passes the pediatrician's review, this agent formats the final, approved recipe into a user-friendly markdown report, including nutrition notes and safety reminders.

**Overall Flow:**

*   The user interacts with the `interactive_recipe_agent`.
*   A basic recipe is generated by `recipe_generator`.
*   The user approves this initial recipe.
*   The `recipe_creation_pipeline` takes over:
    *   The `pediatrician_critic_agent` reviews the recipe.
    *   If it fails, `recipe_refiner_agent` improves it using web research.
    *   This loop continues until the `pediatrician_critic_agent` approves the recipe (or max iterations are reached).
    *   Finally, `final_recipe_presenter_agent` creates a polished markdown report of the approved recipe.

You can edit key parameters (Gemini models, loop iterations) in the `ResearchConfiguration` (if applicable, or other config files like `app/config.py`).

## Customization

You can modify and extend this agent's behavior by editing the backend code.

*   **Modifying Agent Logic:** The core logic for all sub-agents is defined in `app/agent.py`. You can change the prompts, tools, or reasoning steps by modifying the agent definitions here.
*   **Adjusting Parameters:** Key parameters, such as the Gemini models used or the number of refinement loop iterations, can be adjusted in `app/config.py`.
*   **Syncing with Frontend:** The frontend UI integrates with the backend through specific agent names and state keys.
    Important agent/state keys include:
    * `recipe_generator`: Initial recipe creation.
    * `current_recipe`: The state key holding the recipe being worked on.
    * `pediatrician_evaluation`: The state key for the critic's feedback.
    * `final_recipe_report`: The state key for the final markdown output.
    * `interactive_recipe_agent`: Main interaction point.
    
    If you rename agents or change state key names in `app/agent.py`, you must update their references in the frontend code (`/ui`) to maintain functionality.


### Example Interaction

> **User:** Can you make a recipe with sweet potato and carrot?
>
> **Agent:** I can help with that! I've drafted a recipe called "Sunshine Puree" with sweet potato and carrot. It includes simple steps for steaming and blending these veggies. Does this look good? If so, I can have our AI pediatrician review it for safety and nutrition.
>
> **User:** Yes, that sounds great! Please proceed.
>
> *(The agent's `recipe_creation_pipeline` now starts. The `pediatrician_critic_agent` reviews the "Sunshine Puree" recipe.)*
>
> *(After a short while, if the recipe passes the review...)*
>
> **Agent:** Great news! Our AI pediatrician has reviewed the "Sunshine Puree" and it's approved! Here's the final recipe:
>
> # Sunshine Puree
> A delightful and nutritious blend of sweet potato and carrot, perfect for little ones.
> Age Appropriateness: 1-2 years
>
> ## Ingredients
> *   1 medium sweet potato, peeled and cubed
> *   2 medium carrots, peeled and sliced
> *   1/4 cup water (for steaming/blending, adjust as needed)
>
> ## Instructions
> 1.  Steam the sweet potato and carrot cubes/slices until very tender (approx. 15-20 minutes).
> 2.  Allow to cool slightly.
> 3.  Transfer the cooked vegetables to a blender. Add a little water.
> 4.  Blend until smooth, adding more water a tablespoon at a time to reach desired consistency.
>
> ## Nutrition Notes
> This puree is packed with Vitamin A from both sweet potatoes and carrots, which is great for vision and immune health. Sweet potatoes also offer Vitamin C and dietary fiber.
>
> ## Safety First!
> *   Always ensure the puree is cooled to a safe temperature before serving.
> *   Ensure the texture is appropriate for your baby's age and feeding ability to prevent choking. For younger babies, a thinner puree might be needed.

## Troubleshooting

If you encounter issues while setting up or running this agent, here are some resources to help you troubleshoot:
- [ADK Documentation](https://google.github.io/adk-docs/): Comprehensive documentation for the Agent Development Kit
- [Vertex AI Authentication Guide](https://cloud.google.com/vertex-ai/docs/authentication): Detailed instructions for setting up authentication
- [Agent Starter Pack Troubleshooting](https://googlecloudplatform.github.io/agent-starter-pack/guide/troubleshooting.html): Common issues


## 🛠️ Technologies Used

### Backend
*   [**Agent Development Kit (ADK)**](https://github.com/google/adk-python): The core framework for building the stateful, multi-turn agent.
*   [**FastAPI**](https://fastapi.tiangolo.com/): High-performance web framework for the backend API.
*   [**Google Gemini**](https://cloud.google.com/vertex-ai/generative-ai/docs): Used for planning, reasoning, search query generation, and final synthesis.

### Frontend
*   [**React**](https://reactjs.org/) (with [Vite](https://vitejs.dev/)): For building the interactive user interface.
*   [**Tailwind CSS**](https://tailwindcss.com/): For utility-first styling.
*   [**Shadcn UI**](https://ui.shadcn.com/): A set of beautifully designed, accessible components.

## Disclaimer

This agent sample is provided for illustrative purposes only. It serves as a basic example of an agent and a foundational starting point for individuals or teams to develop their own agents.

Users are solely responsible for any further development, testing, security hardening, and deployment of agents based on this sample. We recommend thorough review, testing, and the implementation of appropriate safeguards before using any derived agent in a live or critical system.
