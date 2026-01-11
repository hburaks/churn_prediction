import React from 'react';
import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { cn } from "../../../lib/utils";

interface RiskGaugeProps {
  score: number; // 0 to 100
}

export function RiskGauge({ score }: RiskGaugeProps) {
  const data = [
    {
      name: 'Risk',
      value: score,
      fill: score > 70 ? '#D4183D' : score > 40 ? '#EFF0A3' : '#CFDECA',
    },
  ];

  const riskLabel = score > 70 ? 'Critical' : score > 30 ? 'Moderate' : 'Safe';
  const riskColor = score > 70 ? 'text-destructive' : score > 30 ? 'text-yellow-600' : 'text-green-600';

  return (
    <Card className="h-full flex flex-col items-center justify-center bg-white shadow-sm border-none">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl text-eerie-black/70 font-medium">Churn Probability</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col items-center justify-center relative w-full min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="80%"
            barSize={20}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis
              type="number"
              domain={[0, 100]}
              angleAxisId={0}
              tick={false}
            />
            <RadialBar
              background
              dataKey="value"
              cornerRadius={10}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-8">
          <span className="text-5xl font-bold text-eerie-black">{score}%</span>
          <span className={cn("text-lg font-semibold mt-2 px-3 py-1 rounded-full bg-opacity-20", 
            riskLabel === 'Critical' ? "bg-red-100 text-red-700" : 
            riskLabel === 'Moderate' ? "bg-yellow-100 text-yellow-700" : 
            "bg-green-100 text-green-700"
          )}>
            {riskLabel}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
