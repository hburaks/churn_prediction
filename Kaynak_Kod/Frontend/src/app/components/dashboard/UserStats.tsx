import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { User, CreditCard, Calendar } from "lucide-react";

interface UserStatsProps {
  stats: {
    membershipDays: number;
    totalTransactions: number;
    daysToExpire: number;
  };
}

export function UserStats({ stats }: UserStatsProps) {
  return (
    <Card className="bg-white border-none shadow-sm h-full">
      <CardHeader>
        <CardTitle className="text-xl font-medium text-eerie-black/80">User Stats</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex items-center gap-4 p-4 rounded-xl bg-honeydew/30">
          <div className="p-2 bg-honeydew rounded-full text-green-800">
            <User className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Membership Duration</p>
            <p className="text-xl font-bold text-eerie-black">{stats.membershipDays} Days</p>
          </div>
        </div>

        <div className="flex items-center gap-4 p-4 rounded-xl bg-vanilla/30">
          <div className="p-2 bg-vanilla rounded-full text-yellow-800">
            <CreditCard className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Total Transactions</p>
            <p className="text-xl font-bold text-eerie-black">{stats.totalTransactions}</p>
          </div>
        </div>

        <div className="flex items-center gap-4 p-4 rounded-xl bg-alice-blue/30">
          <div className="p-2 bg-alice-blue rounded-full text-blue-800">
            <Calendar className="w-6 h-6" />
          </div>
          <div>
            <p className="text-sm text-gray-500 font-medium">Subscription Status</p>
            <p className={`text-xl font-bold ${stats.daysToExpire < 0 ? 'text-red-600' : 'text-eerie-black'}`}>
              {stats.daysToExpire < 0 
                ? `Expired (${Math.abs(stats.daysToExpire)} days ago)` 
                : `${stats.daysToExpire} Days Left`}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
