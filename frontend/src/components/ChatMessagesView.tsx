import React from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Copy, CopyCheck } from "lucide-react";
import { InputForm } from "@/components/InputForm";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { ActivityTimeline } from "@/components/ActivityTimeline";
import rehypeRaw from "rehype-raw";
import { API_BASE_URL } from "@/components/constants";
import {mdComponents} from "@/components/MdComponents.tsx";
import {FinalReport} from "@/components/FinalReport.tsx";


interface ProcessedEvent {
  title: string;
  data: never;
}

const urlTransform = (url: string) => {
  if (url.startsWith('data:image/')) {
    return url;
  }
  // For all other URLs, use the default behavior
  const newUrl = new URL(url, 'http://localhost'); // A base URL is required
  return newUrl.href;
};

// Props for HumanMessageBubble
interface HumanMessageBubbleProps {
  message: { content:string; id: string, video?: string, image?: string };
  mdComponents: typeof mdComponents;
}

// HumanMessageBubble Component
const HumanMessageBubble: React.FC<HumanMessageBubbleProps> = ({
  message,
  mdComponents,
}) => {
  return (
    <div className="text-white rounded-3xl break-words min-h-7 bg-neutral-700 max-w-[100%] sm:max-w-[90%] px-4 pt-3 rounded-br-lg">
      <ReactMarkdown components={mdComponents} rehypePlugins={[rehypeRaw]} urlTransform={urlTransform}>
        {message.content}
      </ReactMarkdown>
    </div>
  );
};

// Props for AiMessageBubble
interface AiMessageBubbleProps {
  message: { content: string; id: string; video?: string; image?: string };
  mdComponents: typeof mdComponents;
  handleCopy: (text: string, messageId: string) => void;
  copiedMessageId: string | null;
  agent?: string;
  finalReportContent?: string | boolean; // Updated from finalReportWithCitations
  processedEvents: ProcessedEvent[];
  websiteCount: number;
  isLoading: boolean;
}

// AiMessageBubble Component
const AiMessageBubble: React.FC<AiMessageBubbleProps> = ({
  message,
  mdComponents,
  handleCopy,
  copiedMessageId,
  agent,
  finalReportContent, // Updated prop name
  processedEvents,
  websiteCount,
  isLoading,
}) => {
  const [translatedContent, setTranslatedContent] = useState<string | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);

  const handleTranslate = async () => {
    setIsTranslating(true);
    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: message.content }),
      });
      const data = await response.json();
      setTranslatedContent(data.translated_text);
    } catch (error) {
      console.error("Failed to translate:", error);
    } finally {
      setIsTranslating(false);
    }
  };

  // Show ActivityTimeline if we have processedEvents (this will be the first AI message)
  const shouldShowTimeline = processedEvents.length > 0;
  
  // Condition for DIRECT DISPLAY (interactive_planner_agent OR final report)
  // A final report is identified if finalReportContent is a non-empty string (the report itself)
  // or if it's the old boolean true (though content source is different then).
  // The new agent ('final_recipe_presenter_agent') will provide a string.
  // The old agent ('report_composer_with_citations') might set a boolean.
  const isFinalReport = (typeof finalReportContent === 'string' && finalReportContent.length > 0) ||
                        (typeof finalReportContent === 'boolean' && finalReportContent);

  const shouldDisplayDirectly = 
    agent === "interactive_recipe_agent" || // Changed from interactive_planner_agent
    (agent === "final_recipe_presenter_agent" && isFinalReport) || // New final report agent
    (agent === "image_embedding_agent" && isFinalReport) ||
    (agent === "video_generation_executor" && isFinalReport) ||
    (agent === "image_recipe_agent" && isFinalReport) ||
    (agent === "report_composer_with_citations" && isFinalReport); // Old final report agent

  if (isFinalReport) {
    return (
      <div className="relative break-words flex flex-col w-full">
        <FinalReport
          reportContent={translatedContent || message.content}
          image={message.image}
          video={message.video}
          onTranslate={handleTranslate}
          isTranslating={isTranslating}
        />
      </div>
    );
  }

  if (shouldDisplayDirectly) {
    // Direct display - show content with copy button, and timeline if available
    return (
      <div className="relative break-words flex flex-col w-full">
        {shouldShowTimeline && agent === "interactive_recipe_agent" && (
          <div className="w-full mb-2">
            <ActivityTimeline
              processedEvents={processedEvents}
              isLoading={isLoading}
              websiteCount={websiteCount}
            />
          </div>
        )}
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <ReactMarkdown components={mdComponents} rehypePlugins={[rehypeRaw]} urlTransform={urlTransform}>
              {translatedContent || message.content}
            </ReactMarkdown>
          </div>
          <div className="flex flex-col gap-2">
            <button
              onClick={() => handleCopy(translatedContent || message.content, message.id)}
              className="p-1 hover:bg-neutral-700 rounded"
            >
              {copiedMessageId === message.id ? (
                <CopyCheck className="h-4 w-4 text-green-500" />
              ) : (
                <Copy className="h-4 w-4 text-neutral-400" />
              )}
            </button>
          </div>
        </div>
      </div>
    );
  } else if (shouldShowTimeline) {
    // First AI message with timeline only (no direct content display)
    return (
      <div className="relative break-words flex flex-col w-full">
        <div className="w-full">
          <ActivityTimeline 
            processedEvents={processedEvents}
            isLoading={isLoading}
            websiteCount={websiteCount}
          />
        </div>
        {/* Only show accumulated content if it's not empty and not from research agents */}
        {message.content && message.content.trim() && agent !== "interactive_planner_agent" && (
          <div className="flex items-start gap-3 mt-2">
            <div className="flex-1">
              <ReactMarkdown components={mdComponents} rehypePlugins={[rehypeRaw]} urlTransform={urlTransform}>
                {message.content}
              </ReactMarkdown>
            </div>
            <button
              onClick={() => handleCopy(message.content, message.id)}
              className="p-1 hover:bg-neutral-700 rounded"
            >
              {copiedMessageId === message.id ? (
                <CopyCheck className="h-4 w-4 text-green-500" />
              ) : (
                <Copy className="h-4 w-4 text-neutral-400" />
              )}
            </button>
          </div>
        )}
      </div>
    );
  } else {
    // Fallback for other messages - just show content
    return (
      <div className="relative break-words flex flex-col w-full">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <ReactMarkdown components={mdComponents} rehypePlugins={[rehypeRaw]} urlTransform={urlTransform}>
              {message.content}
            </ReactMarkdown>
          </div>
          <button
            onClick={() => handleCopy(message.content, message.id)}
            className="p-1 hover:bg-neutral-700 rounded"
          >
            {copiedMessageId === message.id ? (
              <CopyCheck className="h-4 w-4 text-green-500" />
            ) : (
              <Copy className="h-4 w-4 text-neutral-400" />
            )}
          </button>
        </div>
      </div>
    );
  }
};

interface ChatMessagesViewProps {
  messages: { type: "human" | "ai"; content: string; id: string; agent?: string; finalReportContent?: string | boolean; video?: string; image?: string }[]; // Updated here
  isLoading: boolean;
  scrollAreaRef: React.RefObject<HTMLDivElement | null>;
  onSubmit: (query: string) => void;
  onImageUpload: (image_b64: string) => void;
  onCancel: () => void;
  displayData: string | null;
  messageEvents: Map<string, ProcessedEvent[]>;
  websiteCount: number;
  inputValue: string;
  setInputValue: (value: string) => void;
}

export function ChatMessagesView({
  messages,
  isLoading,
  scrollAreaRef,
  onSubmit,
  onImageUpload,
  onCancel,
  messageEvents,
  websiteCount,
  inputValue,
  setInputValue,
}: ChatMessagesViewProps) {
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);

  const handleCopy = async (text: string, messageId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error("Failed to copy text:", err);
    }
  };

  const handleNewChat = () => {
    window.location.reload();
  };

  // Find the ID of the last AI message
  const lastAiMessage = messages.slice().reverse().find(m => m.type === "ai");
  const lastAiMessageId = lastAiMessage?.id;

  return (
    <div className="flex flex-col h-full w-full">
      {/* Header with New Chat button */}
      <div className="border-b border-neutral-700 p-4 bg-neutral-800">
        <div className="max-w-4xl mx-auto flex justify-between items-center">
          <h1 className="text-lg font-semibold text-neutral-100">Chat</h1>
          <Button
            onClick={handleNewChat}
            variant="outline"
            className="bg-neutral-700 hover:bg-neutral-600 text-neutral-100 border-neutral-600 hover:border-neutral-500"
          >
            New Chat
          </Button>
        </div>
      </div>
      <div className="flex-1 flex flex-col w-full">
        <ScrollArea ref={scrollAreaRef} className="flex-1 w-full">
          <div className="p-4 md:p-6 space-y-2 max-w-4xl mx-auto">
            {messages.map((message) => { // Removed index as it's not directly used for this logic
              const eventsForMessage = message.type === "ai" ? (messageEvents.get(message.id) || []) : [];
              
              // Determine if the current AI message is the last one
              const isCurrentMessageTheLastAiMessage = message.type === "ai" && message.id === lastAiMessageId;

              return (
                <div
                  key={message.id}
                  className={`flex ${message.type === "human" ? "justify-end" : "justify-start"}`}
                >
                  {message.type === "human" ? (
                    <HumanMessageBubble
                      message={message}
                      mdComponents={mdComponents}
                    />
                  ) : (
                    <AiMessageBubble
                      message={message}
                      mdComponents={mdComponents}
                      handleCopy={handleCopy}
                      copiedMessageId={copiedMessageId}
                      agent={message.agent}
                      finalReportContent={message.finalReportContent} // Updated prop
                      processedEvents={eventsForMessage}
                      // MODIFIED: Pass websiteCount only if it's the last AI message
                      websiteCount={isCurrentMessageTheLastAiMessage ? websiteCount : 0}
                      // MODIFIED: Pass isLoading only if it's the last AI message and global isLoading is true
                      isLoading={isCurrentMessageTheLastAiMessage && isLoading}
                    />
                  )}
                </div>
              );
            })}
            {/* This global "Thinking..." indicator appears below all messages if isLoading is true */}
            {/* It's independent of the per-timeline isLoading state */}
            {isLoading && !lastAiMessage && messages.some(m => m.type === 'human') && (
              <div className="flex justify-start">
                <div className="flex items-center gap-2 text-neutral-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Thinking...</span>
                </div>
              </div>
            )}
             {/* Show "Thinking..." if the last message is human and we are loading, 
                 or if there's an active AI message that is the last one and we are loading.
                 The AiMessageBubble's internal isLoading will handle its own spinner.
                 This one is for the general loading state at the bottom.
             */}
            {isLoading && messages.length > 0 && messages[messages.length -1].type === 'human' && (
                 <div className="flex justify-start pl-10 pt-2"> {/* Adjusted padding to align similarly to AI bubble */}
                    <div className="flex items-center gap-2 text-neutral-400">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Thinking...</span>
                    </div>
                </div>
            )}
          </div>
        </ScrollArea>
      </div>
      <div className="border-t border-neutral-700 p-4 w-full">
        <div className="max-w-3xl mx-auto">
          <InputForm
            onSubmit={onSubmit}
            onImageUpload={onImageUpload}
            isLoading={isLoading}
            context="chat"
            inputValue={inputValue}
            setInputValue={setInputValue}
          />
          {isLoading && (
            <div className="mt-4 flex justify-center">
              <Button
                variant="outline"
                onClick={onCancel}
                className="text-red-400 hover:text-red-300"
              >
                Cancel
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
