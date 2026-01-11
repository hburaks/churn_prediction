import React from 'react';
import { AlertCircle } from 'lucide-react';
import { Button } from "../ui/button";

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center animate-in fade-in-50">
      <div className="bg-red-50 p-4 rounded-full mb-4">
        <AlertCircle className="w-12 h-12 text-red-500" />
      </div>
      <h3 className="text-xl font-semibold text-eerie-black mb-2">Analysis Failed</h3>
      <p className="text-gray-500 max-w-md mb-6">{message}</p>
      <Button 
        onClick={onRetry}
        variant="outline"
        className="border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700"
      >
        Try Again
      </Button>
    </div>
  );
}
