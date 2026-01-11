import React, { useState } from 'react';
import { SearchSection } from './components/dashboard/SearchSection';
import { RiskGauge } from './components/dashboard/RiskGauge';
import { RiskFactors } from './components/dashboard/RiskFactors';
import { UserStats } from './components/dashboard/UserStats';
import { ErrorState } from './components/dashboard/ErrorState';
import { motion } from 'motion/react';
import { Activity } from 'lucide-react';

// Veri Tipleri
interface Reason {
  feature: string;
  value: any;
  impact: string;
  message: string;
}

interface DashboardData {
  riskScore: number;
  riskLevel: string;
  reasons: string[];
  userStats: {
    membershipDays: number;
    totalTransactions: number;
    daysToExpire: number;
  };
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchedId, setSearchedId] = useState('');
  const [error, setError] = useState<string | null>(null);

  const API_BASE = "http://127.0.0.1:8000";

  const handleSearch = async (userId: string) => {
    setLoading(true);
    setSearchedId(userId);
    setData(null);
    setError(null);

    try {
      // 1. Tahmin (Prediction) Al
      const predRes = await fetch(`${API_BASE}/predict/${userId}`);
      if (!predRes.ok) {
        if (predRes.status === 404) throw new Error("User ID not found in the database. Please check the ID and try again.");
        throw new Error("Failed to connect to the prediction engine.");
      }
      const predData = await predRes.json();

      // 2. Açıklama (Explanation) Al
      const expRes = await fetch(`${API_BASE}/explain/${userId}`);
      const expData = await expRes.json();

      // 3. Kullanıcı İstatistiklerini (Stats) Al
      const statsRes = await fetch(`${API_BASE}/user-stats/${userId}`);
      const statsData = await statsRes.json();

      // 4. Verileri Arayüze Uygun Formata Getir
      const formattedData: DashboardData = {
        riskScore: Math.round(predData.churn_probability * 100),
        riskLevel: predData.risk_level,
        reasons: expData.reasons.map((r: Reason) => r.message),
        userStats: {
          membershipDays: statsData.membership_days,
          totalTransactions: statsData.total_transactions,
          daysToExpire: statsData.days_to_expire
        }
      };

      setData(formattedData);
    } catch (err: any) {
      setError(err.message || "An error occurred while fetching data.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-ghost-white p-6 font-urbanist text-eerie-black">
      <header className="mb-10 pt-4 flex flex-col items-center">
        <div className="flex items-center gap-2 mb-2">
          <div className="p-2 bg-eerie-black rounded-lg text-white">
            <Activity className="w-6 h-6" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Churn Prediction</h1>
        </div>
        <p className="text-gray-500">Analyze user retention risk with AI-powered insights</p>
      </header>

      <main className="max-w-6xl mx-auto">
        <SearchSection onSearch={handleSearch} isLoading={loading} />

        {error && (
          <ErrorState message={error} onRetry={() => setError(null)} />
        )}

        {loading && (
          <div className="flex justify-center py-20">
            <div className="animate-pulse flex flex-col items-center">
              <div className="h-4 w-48 bg-gray-200 rounded mb-4"></div>
              <div className="h-64 w-full max-w-4xl bg-gray-200 rounded-xl"></div>
            </div>
          </div>
        )}

        {!loading && !error && data && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="grid grid-cols-1 md:grid-cols-12 gap-6"
          >
            {/* Risk Indicator - Center/Hero */}
            <div className="md:col-span-4 lg:col-span-4 h-full">
              <RiskGauge score={data.riskScore} />
            </div>

            {/* Explainability List */}
            <div className="md:col-span-4 lg:col-span-5 h-full">
              <RiskFactors reasons={data.reasons} />
            </div>

            {/* User Profile Summary */}
            <div className="md:col-span-4 lg:col-span-3 h-full">
              <UserStats stats={data.userStats} />
            </div>
            
            <div className="md:col-span-12 mt-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-alice-blue flex justify-between items-center">
                <div>
                  <h3 className="font-semibold text-lg">Analysis for User ID: <span className="font-mono bg-gray-100 px-2 py-1 rounded">{searchedId}</span></h3>
                  <p className="text-sm text-gray-500">Model Confidence: 94% • Last Updated: Just now</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {!loading && !data && (
          <div className="text-center py-20 text-gray-400">
            <p>Enter a User ID above to see the churn prediction analysis.</p>
          </div>
        )}
      </main>
    </div>
  );
}
