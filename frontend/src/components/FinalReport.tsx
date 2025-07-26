import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";
import {mdComponents} from "@/components/MdComponents.tsx";

interface FinalReportProps {
  reportContent: string;
  image?: string;
  video?: string;
  onTranslate: () => void;
  isTranslating: boolean;
}

export function FinalReport({
  reportContent,
  image,
  video,
  onTranslate,
  isTranslating,
}: FinalReportProps) {
  return (
    <div className="w-full space-y-6">
      {/* Recipe Section */}
      <div className="animate-fadeInUp">
        <h2 className="text-xl font-bold mb-3">🍲 Recipe</h2>
        <div className="animate-fadeInUpSmooth animation-delay-200">
          <ReactMarkdown components={mdComponents} rehypePlugins={[rehypeRaw]}>
            {reportContent}
          </ReactMarkdown>
        </div>
      </div>

      {/* Image Section */}
      {image && (
        <div className="animate-fadeInUp animation-delay-400">
          <h2 className="text-xl font-bold mb-3">🖼️ Recipe Image</h2>
          <div className="animate-fadeInUpSmooth animation-delay-600">
            <img
              src={`data:image/png;base64,${image}`}
              alt="Recipe"
              className="w-full rounded-lg shadow-lg"
            />
          </div>
        </div>
      )}

      {/* Video Section */}
      {video && (
        <div className="animate-fadeInUp animation-delay-800">
          <h2 className="text-xl font-bold mb-3">🎬 Recipe Video</h2>
          <div className="animate-fadeInUpSmooth animation-delay-1000">
            <video className="w-full rounded-lg shadow-lg" controls>
              <source src={video} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-end mt-4 animate-fadeInUp animation-delay-1200">
        <Button
          onClick={onTranslate}
          disabled={isTranslating}
          size="sm"
          variant="outline"
          className="text-xs bg-neutral-800 text-white"
        >
          {isTranslating ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
          {isTranslating ? "Translating..." : "Translate to Portuguese"}
        </Button>
      </div>
    </div>
  );
}