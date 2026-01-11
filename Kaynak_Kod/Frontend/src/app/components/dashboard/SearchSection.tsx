import React, { useState } from 'react';
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Search, Shuffle } from "lucide-react";

interface SearchSectionProps {
  onSearch: (userId: string) => void;
  isLoading: boolean;
}

export function SearchSection({ onSearch, isLoading }: SearchSectionProps) {
  const [userId, setUserId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (userId.trim()) {
      onSearch(userId);
    }
  };

  const handleRandom = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/users/random');
      const data = await res.json();
      if (data && data.user_id !== undefined) {
        const randomId = data.user_id.toString();
        setUserId(randomId);
        onSearch(randomId);
      }
    } catch (error) {
      console.error("Failed to fetch random user:", error);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto mb-8">
      <form onSubmit={handleSubmit} className="flex gap-4 items-center bg-white p-2 rounded-2xl shadow-sm border border-alice-blue">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <Input 
            type="text" 
            placeholder="Enter User ID / MSNO" 
            className="pl-10 border-none shadow-none focus-visible:ring-0 text-lg h-12 bg-transparent"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
          />
        </div>
        <Button 
          type="button"
          variant="secondary"
          onClick={handleRandom}
          disabled={isLoading}
          className="h-12 px-4 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-700"
          title="I'm Feeling Lucky"
        >
          <Shuffle className="w-5 h-5" />
        </Button>
        <Button 
          type="submit" 
          disabled={isLoading}
          className="rounded-xl px-8 h-12 bg-eerie-black text-white hover:bg-gray-800 transition-all font-semibold text-lg"
        >
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </Button>
      </form>
    </div>
  );
}
