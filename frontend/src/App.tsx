import { useState, useRef, useCallback, useEffect } from "react";
import { v4 as uuidv4 } from 'uuid';
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";

// Update DisplayData to be a string type
type DisplayData = string | null;

// Updated to reflect that finalReportContent can be a string
interface MessageWithAgent {
  type: "human" | "ai";
  content: string;
  id: string;
  agent?: string;
  finalReportContent?: string | boolean; // boolean for old logic, string for new
}

interface AgentMessage {
  parts: { text: string }[];
  role: string;
}

interface AgentResponse {
  content: AgentMessage;
  usageMetadata: {
    candidatesTokenCount: number;
    promptTokenCount: number;
    totalTokenCount: number;
  };
  author: string;
  actions: {
    stateDelta: {
      research_plan?: string;
      final_report_with_citations?: boolean;
    };
  };
}

interface ProcessedEvent {
  title: string;
  data: any;
}

export default function App() {
  const [userId, setUserId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [appName, setAppName] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageWithAgent[]>([]);
  const [displayData, setDisplayData] = useState<DisplayData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [messageEvents, setMessageEvents] = useState<Map<string, ProcessedEvent[]>>(new Map());
  const [websiteCount, setWebsiteCount] = useState<number>(0);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [isCheckingBackend, setIsCheckingBackend] = useState(true);
  const currentAgentRef = useRef('');
  const accumulatedTextRef = useRef("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const retryWithBackoff = async (
    fn: () => Promise<any>,
    maxRetries: number = 10,
    maxDuration: number = 120000 // 2 minutes
  ): Promise<any> => {
    const startTime = Date.now();
    let lastError: Error;
    
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      if (Date.now() - startTime > maxDuration) {
        throw new Error(`Retry timeout after ${maxDuration}ms`);
      }
      
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        const delay = Math.min(1000 * Math.pow(2, attempt), 5000); // Exponential backoff, max 5s
        console.log(`Attempt ${attempt + 1} failed, retrying in ${delay}ms...`, error);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError!;
  };

  const createSession = async (): Promise<{userId: string, sessionId: string, appName: string}> => {
    const generatedSessionId = uuidv4();
    const response = await fetch(`/api/apps/app/users/u_999/sessions/${generatedSessionId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return {
      userId: data.userId,
      sessionId: data.id,
      appName: data.appName
    };
  };

  const checkBackendHealth = async (): Promise<boolean> => {
    try {
      // Use the docs endpoint or root endpoint to check if backend is ready
      const response = await fetch("/api/docs", {
        method: "GET",
        headers: {
          "Content-Type": "application/json"
        }
      });
      return response.ok;
    } catch (error) {
      console.log("Backend not ready yet:", error);
      return false;
    }
  };

  // Function to extract text and metadata from SSE data
  const extractDataFromSSE = (data: string) => {
    try {
      const parsed = JSON.parse(data);
      console.log('[SSE PARSED EVENT]:', JSON.stringify(parsed, null, 2)); // DEBUG: Log parsed event

      let textParts: string[] = [];
      let agent = '';
      let finalReportContent: string | boolean | undefined = undefined; // Can be boolean (old) or string (new)
      let functionCall = null;
      let functionResponse = null;
      let sources = null;
      let pediatricianEvaluation = null; // For structured extraction if desired later

      // Check if content.parts exists and has text
      if (parsed.content && parsed.content.parts) {
        textParts = parsed.content.parts
          .filter((part: any) => part.text)
          .map((part: any) => part.text);
        
        // Check for function calls
        const functionCallPart = parsed.content.parts.find((part: any) => part.functionCall);
        if (functionCallPart) {
          functionCall = functionCallPart.functionCall;
        }
        
        // Check for function responses
        const functionResponsePart = parsed.content.parts.find((part: any) => part.functionResponse);
        if (functionResponsePart) {
          functionResponse = functionResponsePart.functionResponse;
        }
      }

      // Extract agent information
      if (parsed.author) {
        agent = parsed.author;
        console.log('[SSE EXTRACT] Agent:', agent); // DEBUG: Log agent
      }

      // Check for new final_recipe_report (string)
      if (parsed.actions?.stateDelta?.final_recipe_report) {
        finalReportContent = parsed.actions.stateDelta.final_recipe_report as string;
        console.log('[SSE EXTRACT] final_recipe_report found:', finalReportContent.substring(0,100));
      }
      // Check for old final_report_with_citations (boolean)
      else if (parsed.actions?.stateDelta?.final_report_with_citations) {
        finalReportContent = parsed.actions.stateDelta.final_report_with_citations as boolean;
        console.log('[SSE EXTRACT] final_report_with_citations found:', finalReportContent);
      }

      // Extract pediatrician_evaluation if present in stateDelta
      if (parsed.actions?.stateDelta?.pediatrician_evaluation) {
        pediatricianEvaluation = parsed.actions.stateDelta.pediatrician_evaluation;
        console.log('[SSE EXTRACT] pediatrician_evaluation found:', pediatricianEvaluation);
      }

      // Extract website count from research agents
      let sourceCount = 0;
      if ((parsed.author === 'section_researcher' || parsed.author === 'enhanced_search_executor')) {
        console.log('[SSE EXTRACT] Relevant agent for source count:', parsed.author); // DEBUG
        if (parsed.actions?.stateDelta?.url_to_short_id) {
          console.log('[SSE EXTRACT] url_to_short_id found:', parsed.actions.stateDelta.url_to_short_id); // DEBUG
          sourceCount = Object.keys(parsed.actions.stateDelta.url_to_short_id).length;
          console.log('[SSE EXTRACT] Calculated sourceCount:', sourceCount); // DEBUG
        } else {
          console.log('[SSE EXTRACT] url_to_short_id NOT found for agent:', parsed.author); // DEBUG
        }
      }

      // Extract sources if available
      if (parsed.actions?.stateDelta?.sources) {
        sources = parsed.actions.stateDelta.sources;
        console.log('[SSE EXTRACT] Sources found:', sources); // DEBUG
      }

      // Return finalReportContent instead of finalReportWithCitations
      return { textParts, agent, finalReportContent, functionCall, functionResponse, sourceCount, sources, pediatricianEvaluation };
    } catch (error) {
      // Log the error and a truncated version of the problematic data for easier debugging.
      const truncatedData = data.length > 200 ? data.substring(0, 200) + "..." : data;
      console.error('Error parsing SSE data. Raw data (truncated): "', truncatedData, '". Error details:', error);
      // Update default return for finalReportContent
      return { textParts: [], agent: '', finalReportContent: undefined, functionCall: null, functionResponse: null, sourceCount: 0, sources: null, pediatricianEvaluation: null };
    }
  };

  // Define getEventTitle here or ensure it's in scope from where it's used
  const getEventTitle = (agentName: string): string => {
    switch (agentName) {
      // Old cases (can be removed if not used by new agent, but keeping for now)
      case "plan_generator":
        return "Planning Research Strategy";
      case "section_planner":
        return "Structuring Report Outline";
      case "section_researcher":
        return "Initial Web Research";
      case "research_evaluator":
        return "Evaluating Research Quality";
      // case "EscalationChecker": // Renamed or handled by new agent name
      //   return "Quality Assessment";
      case "enhanced_search_executor":
        return "Enhanced Web Research";
      case "research_pipeline": // This name is used in the new agent.py
        return "Executing Research Pipeline";
      case "iterative_refinement_loop": // This name is used in the new agent.py
        return "Refining Research";
      // case "interactive_planner_agent": // Renamed or handled by new agent name
      // case "root_agent": // Renamed or handled by new agent name
      //   return "Interactive Planning";

      // New agent names for Tiny Tastes
      case "recipe_generator":
        return "Generating Recipe";
      case "pediatrician_critic_agent":
        return "Pediatrician Review";
      case "escalation_checker": // New agent name from agent.py
        return "Checking Evaluation";
      case "recipe_refiner_agent":
        return "Refining Recipe";
      case "final_recipe_presenter_agent":
        return "Preparing Final Recipe";
      case "interactive_recipe_agent": // This is the new root_agent
        return "Tiny Tastes Assistant";
      case "recipe_creation_pipeline": // This is a sequential agent
        return "Recipe Creation In Progress";
      default:
        return `Processing (${agentName || 'Unknown Agent'})`;
    }
  };

  const processSseEventData = (jsonData: string, aiMessageId: string) => {
    // textParts: from content.parts.text
    // agent: from author
    // finalReportContent: from actions.stateDelta.final_recipe_report (string) or .final_report_with_citations (boolean)
    // functionCall: from content.parts.functionCall
    // functionResponse: from content.parts.functionResponse
    // sourceCount: from actions.stateDelta.url_to_short_id (used by old research agents)
    // sources: from actions.stateDelta.sources (used by old research agents)
    // pediatricianEvaluation: from actions.stateDelta.pediatrician_evaluation
    const { textParts, agent, finalReportContent, functionCall, functionResponse, sourceCount, sources, pediatricianEvaluation } = extractDataFromSSE(jsonData);

    // The new agent.py structure:
    // `interactive_recipe_agent` (root) -> text updates the main AI message. It calls:
    //    `recipe_generator` (tool) -> its output `current_recipe` (JSON) is handled by `interactive_recipe_agent` to present to user.
    //    `recipe_creation_pipeline` (sequential agent) -> delegates to:
    //        `iterative_refinement_loop` (loop agent) -> delegates to:
    //            `pediatrician_critic_agent` -> output `pediatrician_evaluation` (JSON with grade, comment, follow_up_questions)
    //            `escalation_checker` -> checks grade, may escalate to stop loop.
    //            `recipe_refiner_agent` -> uses google_search tool, output `current_recipe` (JSON)
    //        `final_recipe_presenter_agent` -> output `final_recipe_report` (Markdown string)

    // It's important to get the `final_recipe_report` from the state delta when `final_recipe_presenter_agent` runs.
    // Let's assume `final_report_with_citations` in `extractDataFromSSE` can be adapted or a new field like `final_recipe_report` is checked.
    // For now, I'll check if `parsed.actions?.stateDelta?.final_recipe_report` exists in `extractDataFromSSE`.

    // Let's refine extractDataFromSSE to look for `final_recipe_report`
    // This requires modifying extractDataFromSSE. I'll do that after this block.

    if (sourceCount > 0) { // Keep for now, in case any tool still uses url_to_short_id
      console.log('[SSE HANDLER] Updating websiteCount. Current sourceCount:', sourceCount);
      setWebsiteCount(prev => Math.max(prev, sourceCount));
    }

    if (agent && agent !== currentAgentRef.current) {
      currentAgentRef.current = agent;
    }

    if (functionCall) {
      const functionCallTitle = `Function Call: ${functionCall.name}`;
      console.log('[SSE HANDLER] Adding Function Call timeline event:', functionCallTitle, 'Args:', functionCall.args);
      setMessageEvents(prev => new Map(prev).set(aiMessageId, [...(prev.get(aiMessageId) || []), {
        title: functionCallTitle,
        data: { type: 'functionCall', name: functionCall.name, args: functionCall.args, id: functionCall.id }
      }]));

      // Special handling for recipe_generator tool output if needed for timeline
      if (functionCall.name === 'recipe_generator') {
        // The actual recipe content will be in the functionResponse.
        // The interactive_recipe_agent will then use this to update its text.
      }
    }

    if (functionResponse) {
      const functionResponseTitle = `Function Response: ${functionResponse.name}`;
      console.log('[SSE HANDLER] Adding Function Response timeline event:', functionResponseTitle);
      setMessageEvents(prev => new Map(prev).set(aiMessageId, [...(prev.get(aiMessageId) || []), {
        title: functionResponseTitle,
        // Note: functionResponse.response might be large (e.g. full recipe JSON). Consider if it should be displayed in timeline.
        data: { type: 'functionResponse', name: functionResponse.name, response: functionResponse.response, id: functionResponse.id }
      }]));

      // If the function response is from recipe_generator, the interactive_recipe_agent should then speak about it.
      // The text from interactive_recipe_agent (which includes details from current_recipe) will update the main message.
    }

    // Handle text parts
    // `interactive_recipe_agent` updates the main chat.
    // Other agents primarily contribute to the timeline.
    if (textParts.length > 0) {
      if (agent === "interactive_recipe_agent") { // This agent's text updates the main AI message
        for (const text of textParts) {
          accumulatedTextRef.current += text; // Removed extra space, let LLM decide spacing.
          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId ? { ...msg, content: accumulatedTextRef.current.trim(), agent: currentAgentRef.current || msg.agent } : msg
          ));
          setDisplayData(accumulatedTextRef.current.trim());
        }
      } else if (agent !== "final_recipe_presenter_agent") { // Text from other agents goes to timeline (unless it's the final report)
        const eventTitle = getEventTitle(agent);
        const combinedText = textParts.join(" ");
        // Avoid adding empty text events to timeline
        if (combinedText.trim()) {
          console.log('[SSE HANDLER] Adding Text timeline event for agent:', agent, 'Title:', eventTitle, 'Data:', combinedText);
          setMessageEvents(prev => new Map(prev).set(aiMessageId, [...(prev.get(aiMessageId) || []), {
            title: eventTitle,
            data: { type: 'text', content: combinedText }
          }]));
        }
      }
    }

    if (sources) { // Keep for google_search tool if it populates this
      console.log('[SSE HANDLER] Adding Retrieved Sources timeline event:', sources);
      setMessageEvents(prev => new Map(prev).set(aiMessageId, [...(prev.get(aiMessageId) || []), {
        title: "Retrieved Sources", data: { type: 'sources', content: sources }
      }]));
    }

    // Check for pediatrician_evaluation to add to timeline
    // Use pediatricianEvaluation extracted by extractDataFromSSE
    if (pediatricianEvaluation && agent === "pediatrician_critic_agent") {
        console.log('[SSE HANDLER] Adding Pediatrician Feedback timeline event:', pediatricianEvaluation);
        let displayEvaluation = `Grade: ${pediatricianEvaluation.grade}. Comment: ${pediatricianEvaluation.comment}`;
        if (pediatricianEvaluation.follow_up_questions && pediatricianEvaluation.follow_up_questions.length > 0) {
            displayEvaluation += ` Follow-up: ${pediatricianEvaluation.follow_up_questions.map((q: any) => q.question).join(', ')}`;
        }
        setMessageEvents(prev => new Map(prev).set(aiMessageId, [...(prev.get(aiMessageId) || []), {
          title: "Pediatrician Feedback",
          data: { type: 'text', content: displayEvaluation }
        }]));
    }

    // Handle final report content (could be new recipe string or old boolean)
    if (finalReportContent) {
      if (agent === "final_recipe_presenter_agent" && typeof finalReportContent === 'string') {
        console.log('[SSE HANDLER] Final recipe report received from final_recipe_presenter_agent.');
        setMessages(prev => {
          const filtered = prev.filter(msg => msg.id !== aiMessageId || msg.content.trim() !== "");
          return [...filtered, {
            type: "ai",
            content: finalReportContent as string, // Is a string here
            id: aiMessageId + "_final",
            agent: currentAgentRef.current,
            finalReportContent: finalReportContent // Store the string content
          }];
        });
        setDisplayData(finalReportContent as string);
      } else if (agent === "report_composer_with_citations" && typeof finalReportContent === 'boolean' && finalReportContent === true) {
        // This case handles the old boolean `final_report_with_citations`.
        // However, the content for the message is not directly available in `finalReportContent` if it's just a boolean.
        // The old code used `finalReportWithCitations as string` which was problematic.
        // This part of the logic might be dead code if `report_composer_with_citations` is no longer used or if it now sends a string.
        // For safety, if it's a boolean, we might need a placeholder or to ensure the text content is already in `accumulatedTextRef.current`.
        // The original code created a new message with `finalReportWithCitations as string`.
        // Let's assume if finalReportContent is boolean true, the actual content is already streamed via textParts from report_composer_with_citations.
        // The original logic was:
        // if (agent === "report_composer_with_citations" && finalReportWithCitations) {
        //   const finalReportMessageId = Date.now().toString() + "_final";
        //   setMessages(prev => [...prev, { type: "ai", content: finalReportWithCitations as string, id: finalReportMessageId, agent: currentAgentRef.current, finalReportWithCitations: true }]);
        //   setDisplayData(finalReportWithCitations as string);
        // }
        // This implies `finalReportWithCitations` (the boolean) was being cast to string for content, which is wrong.
        // A more correct handling for the old system would be to use the accumulated text for that agent if finalReportWithCitations (boolean) is true.
        // However, since the new agent sends a string, this path might not be hit.
        // If it is hit, and `finalReportContent` is `true` (boolean), we should use `accumulatedTextRef.current`.
        console.warn('[SSE HANDLER] Old final report boolean flag received. Content should have been streamed.');
        // The accumulatedTextRef should hold the content from report_composer_with_citations if it streamed text.
        // The message update for this agent's text should ideally handle making it final.
        // Let's ensure the message is marked as final.
        setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId ? { ...msg, finalReportContent: true, agent: currentAgentRef.current } : msg
        ));
        // setDisplayData might have already been set by the text stream.
      }
    }
  };

  const handleSubmit = useCallback(async (query: string, model: string, effort: string) => {
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      // Create session if it doesn't exist
      let currentUserId = userId;
      let currentSessionId = sessionId;
      let currentAppName = appName;
      
      if (!currentSessionId || !currentUserId || !currentAppName) {
        console.log('Creating new session...');
        const sessionData = await retryWithBackoff(createSession);
        currentUserId = sessionData.userId;
        currentSessionId = sessionData.sessionId;
        currentAppName = sessionData.appName;
        
        setUserId(currentUserId);
        setSessionId(currentSessionId);
        setAppName(currentAppName);
        console.log('Session created successfully:', { currentUserId, currentSessionId, currentAppName });
      }

      // Add user message to chat
      const userMessageId = Date.now().toString();
      setMessages(prev => [...prev, { type: "human", content: query, id: userMessageId }]);

      // Create AI message placeholder
      const aiMessageId = Date.now().toString() + "_ai";
      currentAgentRef.current = ''; // Reset current agent
      accumulatedTextRef.current = ''; // Reset accumulated text

      setMessages(prev => [...prev, {
        type: "ai",
        content: "",
        id: aiMessageId,
        agent: '',
      }]);

      // Send the message with retry logic
      const sendMessage = async () => {
        const response = await fetch("/api/run_sse", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            appName: currentAppName,
            userId: currentUserId,
            sessionId: currentSessionId,
            newMessage: {
              parts: [{ text: query }],
              role: "user"
            },
            streaming: false
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to send message: ${response.status} ${response.statusText}`);
        }
        
        return response;
      };

      const response = await retryWithBackoff(sendMessage);

      // Handle SSE streaming
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = ""; 
      let eventDataBuffer = "";

      if (reader) {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();

          if (value) {
            lineBuffer += decoder.decode(value, { stream: true });
          }
          
          let eolIndex;
          // Process all complete lines in the buffer, or the remaining buffer if 'done'
          while ((eolIndex = lineBuffer.indexOf('\n')) >= 0 || (done && lineBuffer.length > 0)) {
            let line: string;
            if (eolIndex >= 0) {
              line = lineBuffer.substring(0, eolIndex);
              lineBuffer = lineBuffer.substring(eolIndex + 1);
            } else { // Only if done and lineBuffer has content without a trailing newline
              line = lineBuffer;
              lineBuffer = "";
            }

            if (line.trim() === "") { // Empty line: dispatch event
              if (eventDataBuffer.length > 0) {
                // Remove trailing newline before parsing
                const jsonDataToParse = eventDataBuffer.endsWith('\n') ? eventDataBuffer.slice(0, -1) : eventDataBuffer;
                console.log('[SSE DISPATCH EVENT]:', jsonDataToParse.substring(0, 200) + "..."); // DEBUG
                processSseEventData(jsonDataToParse, aiMessageId);
                eventDataBuffer = ""; // Reset for next event
              }
            } else if (line.startsWith('data:')) {
              eventDataBuffer += line.substring(5).trimStart() + '\n'; // Add newline as per spec for multi-line data
            } else if (line.startsWith(':')) {
              // Comment line, ignore
            } // Other SSE fields (event, id, retry) can be handled here if needed
          }

          if (done) {
            // If the loop exited due to 'done', and there's still data in eventDataBuffer
            // (e.g., stream ended after data lines but before an empty line delimiter)
            if (eventDataBuffer.length > 0) {
              const jsonDataToParse = eventDataBuffer.endsWith('\n') ? eventDataBuffer.slice(0, -1) : eventDataBuffer;
              console.log('[SSE DISPATCH FINAL EVENT]:', jsonDataToParse.substring(0,200) + "..."); // DEBUG
              processSseEventData(jsonDataToParse, aiMessageId);
              eventDataBuffer = ""; // Clear buffer
            }
            break; // Exit the while(true) loop
          }
        }
      }

      setIsLoading(false);

    } catch (error) {
      console.error("Error:", error);
      // Update the AI message placeholder with an error message
      const aiMessageId = Date.now().toString() + "_ai_error";
      setMessages(prev => [...prev, { 
        type: "ai", 
        content: `Sorry, there was an error processing your request: ${error instanceof Error ? error.message : 'Unknown error'}`, 
        id: aiMessageId 
      }]);
      setIsLoading(false);
    }
  }, [processSseEventData]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [messages]);

  useEffect(() => {
    const checkBackend = async () => {
      setIsCheckingBackend(true);
      
      // Check if backend is ready with retry logic
      const maxAttempts = 60; // 2 minutes with 2-second intervals
      let attempts = 0;
      
      while (attempts < maxAttempts) {
        const isReady = await checkBackendHealth();
        if (isReady) {
          setIsBackendReady(true);
          setIsCheckingBackend(false);
          return;
        }
        
        attempts++;
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds between checks
      }
      
      // If we get here, backend didn't come up in time
      setIsCheckingBackend(false);
      console.error("Backend failed to start within 2 minutes");
    };
    
    checkBackend();
  }, []);

  const handleCancel = useCallback(() => {
    setMessages([]);
    setDisplayData(null);
    setMessageEvents(new Map());
    setWebsiteCount(0);
    window.location.reload();
  }, []);

  // Scroll to bottom when messages update
  const scrollToBottom = useCallback(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, []);

  const BackendLoadingScreen = () => (
    <div className="flex-1 flex flex-col items-center justify-center p-4 overflow-hidden relative">
      <div className="w-full max-w-2xl z-10
                      bg-neutral-900/50 backdrop-blur-md 
                      p-8 rounded-2xl border border-neutral-700 
                      shadow-2xl shadow-black/60">
        
        <div className="text-center space-y-6">
          <h1 className="text-4xl font-bold text-white flex items-center justify-center gap-3">
            ✨ Tiny Tastes - Baby Recipe Planner 🍲
          </h1>
          
          <div className="flex flex-col items-center space-y-4">
            {/* Spinning animation */}
            <div className="relative">
              <div className="w-16 h-16 border-4 border-neutral-600 border-t-blue-500 rounded-full animate-spin"></div>
              <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-r-purple-500 rounded-full animate-spin" style={{animationDirection: 'reverse', animationDuration: '1.5s'}}></div>
            </div>
            
            <div className="space-y-2">
              <p className="text-xl text-neutral-300">
                Waiting for backend to be ready...
              </p>
              <p className="text-sm text-neutral-400">
                This may take a moment on first startup
              </p>
            </div>
            
            {/* Animated dots */}
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
              <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-neutral-800 text-neutral-100 font-sans antialiased">
      <main className="flex-1 flex flex-col overflow-hidden w-full">
        <div className={`flex-1 overflow-y-auto ${(messages.length === 0 || isCheckingBackend) ? "flex" : ""}`}>
          {isCheckingBackend ? (
            <BackendLoadingScreen />
          ) : !isBackendReady ? (
            <div className="flex-1 flex flex-col items-center justify-center p-4">
              <div className="text-center space-y-4">
                <h2 className="text-2xl font-bold text-red-400">Backend Unavailable</h2>
                <p className="text-neutral-300">
                  Unable to connect to backend services at localhost:8000
                </p>
                <button 
                  onClick={() => window.location.reload()} 
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={isLoading}
              onCancel={handleCancel}
            />
          ) : (
            <ChatMessagesView
              messages={messages}
              isLoading={isLoading}
              scrollAreaRef={scrollAreaRef}
              onSubmit={handleSubmit}
              onCancel={handleCancel}
              displayData={displayData}
              messageEvents={messageEvents}
              websiteCount={websiteCount}
            />
          )}
        </div>
      </main>
    </div>
  );
}
