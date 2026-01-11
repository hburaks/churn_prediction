import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { AlertTriangle, TrendingDown, Clock } from "lucide-react";

interface RiskFactorsProps {
  reasons: string[];
}

export function RiskFactors({ reasons }: RiskFactorsProps) {
  return (
    <Card className="bg-white border-none shadow-sm h-full">
      <CardHeader>
        <CardTitle className="text-xl font-medium text-eerie-black/80">Risk Factors</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-4">
          {reasons.map((reason, index) => (
            <li key={index} className="flex items-start gap-3 p-3 rounded-lg bg-alice-blue/30 hover:bg-alice-blue/50 transition-colors">
              <div className="mt-1 min-w-[20px]">
                {index === 0 ? <AlertTriangle className="w-5 h-5 text-red-500" /> :
                 index === 1 ? <TrendingDown className="w-5 h-5 text-orange-500" /> :
                 <Clock className="w-5 h-5 text-blue-500" />}
              </div>
              <span className="text-eerie-black font-medium text-base">{reason}</span>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
