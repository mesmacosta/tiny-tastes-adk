import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Send, Camera } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";

interface InputFormProps {
  onSubmit: (query: string) => void;
  onImageUpload: (image_b64: string) => void;
  isLoading: boolean;
  context?: 'homepage' | 'chat';
  inputValue: string;
  setInputValue: (value: string) => void;
}

export function InputForm({ onSubmit, onImageUpload, isLoading, context = 'homepage', inputValue, setInputValue }: InputFormProps) {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraOpen(true);
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    setIsCameraOpen(false);
    setCapturedImage(null);
  };

  const handleCapture = () => {
    const canvas = document.createElement("canvas");
    if (videoRef.current) {
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg");
        setCapturedImage(dataUrl);
      }
    }
  };

  const handleSendImage = () => {
    if (capturedImage) {
      const base64Image = capturedImage.split(",")[1];
      onImageUpload(base64Image);
      stopCamera();
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSubmit(inputValue.trim());
      setInputValue("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const placeholderText =
    context === 'chat'
      ? "Type 'Looks good' to approve, or provide feedback..."
      : "Enter ingredients... e.g., avocado, banana, sweet potato";

  return (
    <>
      <form onSubmit={handleSubmit} className="flex flex-col gap-2">
        <div className="flex items-end space-x-2">
          <Textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholderText}
            rows={1}
            className="flex-1 resize-none pr-10 min-h-[40px]"
          />
          <Button type="button" size="icon" onClick={startCamera} disabled={isLoading}>
            <Camera className="h-4 w-4" />
          </Button>
          <Button type="submit" size="icon" disabled={isLoading || !inputValue.trim()}>
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
      </form>

      <Dialog open={isCameraOpen} onOpenChange={setIsCameraOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Capture Ingredients</DialogTitle>
          </DialogHeader>
          <div className="flex justify-center">
            {capturedImage ? (
              <img src={capturedImage} alt="Captured" />
            ) : (
              <video ref={videoRef} autoPlay playsInline />
            )}
          </div>
          <DialogFooter>
            {capturedImage ? (
              <>
                <Button onClick={() => setCapturedImage(null)}>Retake</Button>
                <Button onClick={handleSendImage}>Send</Button>
              </>
            ) : (
              <Button onClick={handleCapture}>Capture</Button>
            )}
            <Button variant="outline" onClick={stopCamera}>Cancel</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
