import React, { Suspense, lazy } from 'react';
import { Eye, Brain, MessageCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import LoadingSpinner from '../components/LoadingSpinner'; // Create a simple spinner component

// Lazy load components for better performance
const FeatureCard = lazy(() => import('./FeatureCard'));

const HomePage = () => {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], ['0%', '50%']);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="relative h-[700px] overflow-hidden">
        {/* Background Video */}
        <video
          className="absolute inset-0 w-full h-full object-cover"
          autoPlay
          loop
          muted
          playsInline
        >
          <source src="https://media.istockphoto.com/id/1167979998/video/extreme-closeup-on-blue-eye-entering-human-mind.mp4?s=mp4-640x640-is&k=20&c=Oiauo-LfMTBSfyQaghddYSphMCkjla6SBe5B_mRMeFM=" type="video/mp4" />
          Your browser does not support the video tag.
        </video>

        {/* Gradient Overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/70 to-indigo-900/70"></div>

        <div className="relative z-10 h-full flex flex-col items-center justify-center text-white px-4">
          <motion.h1
            className="text-5xl md:text-7xl font-bold mb-6 text-center"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.5 }}
          >
            Advanced OCT Analysis
          </motion.h1>
          <motion.p
            className="text-xl md:text-2xl mb-12 text-center max-w-3xl"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1 }}
          >
            AI-Powered Retinal Disease Detection for Accurate and Fast Diagnoses
          </motion.p>
          <motion.div
            className="flex flex-wrap gap-6 justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.5 }}
          >
            <Link
              to="/classify"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg flex items-center text-lg transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-lg"
              aria-label="Start Analysis"
            >
              <Eye className="mr-2" /> Start Analysis
            </Link>
          </motion.div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 px-4 bg-white">
        <div className="max-w-6xl mx-auto">
          <motion.h2
            className="text-4xl font-bold text-center mb-12"
            initial={{ opacity: 0, y: -20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            viewport={{ once: true }}
          >
            Our Technology
          </motion.h2>
          <div className="grid md:grid-cols-3 gap-8">
            <Suspense fallback={<LoadingSpinner />}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                viewport={{ once: true }}
              >
                <FeatureCard
                  icon={<Brain className="w-12 h-12 text-blue-600" />}
                  title="AI Detection"
                  description="State-of-the-art deep learning models for accurate and reliable diagnosis."
                />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                viewport={{ once: true }}
              >
                <FeatureCard
                  icon={<Eye className="w-12 h-12 text-blue-600" />}
                  title="Multiple Models"
                  description="Support for various architectures including MobileNet, ResNet, and EfficientNet."
                />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
                viewport={{ once: true }}
              >
                <FeatureCard
                  icon={<MessageCircle className="w-12 h-12 text-blue-600" />}
                  title="Interactive Chat"
                  description="Get instant answers about conditions, treatments, and results."
                />
              </motion.div>
            </Suspense>
          </div>
        </div>
      </div>

      {/* Gradient CTA Section */}
      <div className="relative py-20 bg-gradient-to-r from-blue-900 to-indigo-900 overflow-hidden">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-blue-800/50 to-indigo-800/50"
          initial={{ scale: 1.2 }}
          animate={{ scale: 1 }}
          transition={{ duration: 2, ease: 'easeInOut' }}
        />
        <div className="relative z-10 max-w-6xl mx-auto px-4 text-center">
          <motion.h2
            className="text-4xl font-bold text-white mb-6"
            initial={{ opacity: 0, y: -20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            viewport={{ once: true }}
          >
            Ready to Transform Retinal Diagnostics?
          </motion.h2>
          <motion.p
            className="text-xl text-white mb-12 max-w-2xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
            viewport={{ once: true }}
          >
            Join thousands of medical professionals leveraging AI for faster, more accurate diagnoses.
          </motion.p>
          <motion.div
            className="flex justify-center"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
            viewport={{ once: true }}
          >
            <Link
              to="/classify"
              className="bg-white text-blue-900 px-8 py-3 rounded-lg flex items-center text-lg font-semibold transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-lg"
              aria-label="Get Started"
            >
              <Eye className="mr-2" /> Get Started
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;