import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const formatVND = (value) => {
  return new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND' }).format(value);
};

const ShapChart = ({ data, title }) => {
  if (!data || !data.top_drivers || !data.top_barriers) return null;

  // Gộp drivers và barriers lại để vẽ chung 1 biểu đồ
  const drivers = data.top_drivers.map(d => ({ name: d.feature, impact: d.impact, type: 'driver' }));
  const barriers = data.top_barriers.map(d => ({ name: d.feature, impact: d.impact, type: 'barrier' }));
  
  const chartData = [...drivers, ...barriers];

  // Không vẽ biểu đồ nếu không có dữ liệu nào đáng kể
  if (chartData.length === 0) return null;

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 mt-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{title}</h3>
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <XAxis type="number" hide />
            <YAxis dataKey="name" type="category" width={80} tick={{fontSize: 11, fill: '#4b5563'}} />
            <Tooltip formatter={(value) => formatVND(value)} />
            <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.type === 'driver' ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-4 text-xs mt-2 justify-center text-gray-500">
        <div className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-emerald-500"></span> Động lực tăng</div>
        <div className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500"></span> Rào cản giảm</div>
      </div>
    </div>
  );
};

export default ShapChart;
