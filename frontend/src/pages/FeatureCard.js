import React from 'react';
import { motion } from 'framer-motion';

const FeatureCard = ({ icon, title, description }) => (
  <motion.div
    className="p-6 border rounded-lg shadow-lg hover:shadow-xl transition-shadow duration-300 ease-in-out bg-white"
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
  >
    <div className="w-12 h-12 mb-4">
      {icon}
    </div>
    <h3 className="text-xl font-bold mb-2">{title}</h3>
    <p className="text-gray-600">{description}</p>
  </motion.div>
);

export default FeatureCard;