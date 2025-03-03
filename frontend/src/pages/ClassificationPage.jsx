import React, { useState, useCallback } from 'react';
import { Upload, Info, RefreshCw, ChevronRight, ChevronLeft, Eye, EyeOff, ZoomIn, ZoomOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import OCTClassifier from '../components/classification/OCTClassifier';
import GradCAMVisual from '../components/classification/GradCAMVisualization';
import { classifyImage } from '../utils/api';
import Chatbot from '../components/chat/ChatBot';

const ClassificationPage = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('mobilenet_v3');
  const [error, setError] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showUploadSection, setShowUploadSection] = useState(true);

  const models = [
    { value: 'mobilenet_v3', label: 'MobileNet V3', description: 'Lightweight and fast, ideal for mobile devices.', icon: '📱' },
    { value: 'efficientnet_b0', label: 'EfficientNet B0', description: 'Balanced accuracy and efficiency.', icon: '⚖️' },
    { value: 'resnet18', label: 'ResNet 18', description: 'Deep residual learning for image recognition.', icon: '🏗️' },
    { value: 'resnet50', label: 'ResNet 50', description: 'Improved accuracy with deeper architecture.', icon: '🏢' },
    { value: 'squeezenet1_1', label: 'SqueezeNet', description: 'Compact model with minimal parameters.', icon: '🎯' },
    { value: 'mobilevit_s', label: 'MobileViT', description: 'Combines CNNs and Transformers for mobile devices.', icon: '🤖' },
  ];

  const handleImageUpload = async (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file.');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError('File size should be less than 5MB.');
      return;
    }

    setSelectedImage(URL.createObjectURL(file));
    setIsLoading(true);
    setError(null);
    setGradcamImage(null);
    setShowGradCAM(false);
    setShowUploadSection(false);

    try {
      const result = await classifyImage(file, selectedModel);
      setPrediction(result);

      if (result.gradcam) {
        const gradcamUrl = result.gradcam.startsWith('data:image')
          ? result.gradcam
          : `data:image/png;base64,${result.gradcam}`;
        setGradcamImage(gradcamUrl);
      }
    } catch (error) {
      console.error('Error:', error);
      setError(`Error processing image: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPrediction(null);
    setError(null);
    setGradcamImage(null);
    setShowGradCAM(false);
    setZoomLevel(1);
    setShowUploadSection(true);
  };

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const toggleGradCAM = () => {
    setShowGradCAM(!showGradCAM);
  };

  const handleZoomIn = () => {
    setZoomLevel((prev) => Math.min(prev + 0.1, 2));
  };

  const handleZoomOut = () => {
    setZoomLevel((prev) => Math.max(prev - 0.1, 1));
  };

  const onDrop = useCallback((acceptedFiles) => {
    handleImageUpload(acceptedFiles[0]);
  }, []);

  return (
    <div
      className="relative bg-cover bg-center min-h-screen"
      style={{
        backgroundImage:
          "url('https://static.vecteezy.com/system/resources/thumbnails/032/838/286/small_2x/blue-eye-staring-reflecting-nature-abstract-beauty-generated-by-ai-photo.jpg')",
      }}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-black/70 to-black/50"></div>

      <div className="relative max-w-5xl mx-auto p-8 text-white">
        <motion.h1
          className="text-5xl font-extrabold mb-8 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          OCT Image Classification
        </motion.h1>

        <motion.div
          className="mb-12 bg-white/10 backdrop-blur-sm rounded-lg p-6 shadow-2xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <label className="block text-lg font-semibold mb-2">Select Model</label>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {models.map((model) => (
              <motion.div
                key={model.value}
                className={`p-4 rounded-lg cursor-pointer transition-all duration-300 ${
                  selectedModel === model.value
                    ? 'bg-blue-600/50 border-blue-400'
                    : 'bg-white/10 hover:bg-white/20'
                }`}
                onClick={() => setSelectedModel(model.value)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-xl">{model.icon}</span>
                  <span className="font-semibold">{model.label}</span>
                </div>
                <p className="text-sm text-gray-300 mt-2">{model.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <AnimatePresence>
          {showUploadSection && (
            <motion.div
              className="mb-12 bg-white/10 backdrop-blur-sm rounded-lg p-6 shadow-2xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <label className="block text-lg font-semibold mb-2">Upload OCT Image</label>
              <motion.div
                className="border-2 border-dashed border-blue-400 rounded-lg p-8 text-center bg-white/10 hover:border-blue-600 transition-all duration-300 relative"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const file = e.dataTransfer.files[0];
                  handleImageUpload(file);
                }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(e.target.files[0])}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload" className="cursor-pointer">
                  <motion.div
                    className="flex flex-col items-center justify-center"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Upload className="h-12 w-12 text-blue-400 mb-3" />
                    <p className="text-lg font-semibold text-blue-400">Click or drag image to upload</p>
                  </motion.div>
                </label>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            className="text-center py-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="animate-spin rounded-full h-10 w-10 border-b-4 border-blue-600 mx-auto"></div>
            <p className="mt-2 text-blue-400 font-medium">Processing...</p>
          </motion.div>
        )}

        {error && (
          <motion.div
            className="bg-red-200 border border-red-500 text-red-800 px-4 py-3 rounded-lg mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            {error}
          </motion.div>
        )}

        {selectedImage && !isLoading && (
          <motion.div
            className="grid md:grid-cols-2 gap-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div>
              <h3 className="text-lg font-bold text-blue-400 mb-2">Original Image</h3>
              <div className="relative h-64 w-full rounded-lg overflow-hidden">
                <img
                  src={selectedImage}
                  alt="Uploaded OCT"
                  className="w-full h-full object-cover"
                  style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }}
                />
                <div className="absolute bottom-4 right-4 flex space-x-2">
                  <button
                    onClick={handleZoomIn}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full"
                  >
                    <ZoomIn className="h-5 w-5" />
                  </button>
                  <button
                    onClick={handleZoomOut}
                    className="bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-full"
                  >
                    <ZoomOut className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
            {prediction && (
              <div className="bg-white/10 backdrop-blur-sm p-6 rounded-lg shadow-md border border-gray-200">
                <h3 className="text-lg font-bold text-blue-400 mb-4">Results</h3>
                <div className="space-y-3">
                  <p className="text-gray-300">
                    <span className="font-semibold text-blue-400">Condition:</span>{' '}
                    <span className="text-blue-300">{prediction.class}</span>
                  </p>
                  <p className="text-gray-300">
                    <span className="font-semibold text-blue-400">Confidence:</span>{' '}
                    <span className="text-blue-300">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </p>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {gradcamImage && !isLoading && (
          <motion.div
            className="mt-8 bg-white/10 backdrop-blur-sm rounded-lg p-6 shadow-2xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-bold text-blue-400">Grad-CAM Visualization</h3>
              <button
                onClick={toggleGradCAM}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center justify-center transition-all duration-300"
              >
                {showGradCAM ? <EyeOff className="mr-2" /> : <Eye className="mr-2" />}
                {showGradCAM ? 'Hide Grad-CAM' : 'Show Grad-CAM'}
              </button>
            </div>
            <AnimatePresence>
              {showGradCAM && (
                <motion.div
                  className="grid grid-cols-1 md:grid-cols-3 gap-4"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div>
                    <h4 className="text-md font-semibold text-blue-400 mb-2">Original Image</h4>
                    <img
                      src={selectedImage}
                      alt="Original OCT"
                      className="w-full h-48 object-cover rounded-lg"
                    />
                  </div>
                  <div>
                    <h4 className="text-md font-semibold text-blue-400 mb-2">Grad-CAM Heatmap</h4>
                    <img
                      src={gradcamImage}
                      alt="Grad-CAM Heatmap"
                      className="w-full h-48 object-cover rounded-lg"
                    />
                  </div>
                  <div>
                    <h4 className="text-md font-semibold text-blue-400 mb-2">Overlay</h4>
                    <div className="relative h-48 w-full rounded-lg overflow-hidden">
                      <img
                        src={selectedImage}
                        alt="Original OCT"
                        className="w-full h-full object-cover"
                      />
                      <img
                        src={gradcamImage}
                        alt="Grad-CAM Overlay"
                        className="absolute inset-0 w-full h-full object-cover opacity-50"
                        style={{ opacity: overlayOpacity }}
                      />
                    </div>
                    <div className="mt-2">
                      <label className="block text-sm font-medium text-blue-400">Overlay Opacity</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={overlayOpacity}
                        onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                        className="w-full"
                      />
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {selectedImage && !isLoading && (
          <motion.div
            className="text-center mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <button
              onClick={handleReset}
              className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-2 rounded-lg flex items-center justify-center mx-auto transition-all duration-300"
            >
              <RefreshCw className="mr-2" /> Reset
            </button>
          </motion.div>
        )}
      </div>

      <div className="fixed bottom-12 right-8 z-50">
        <motion.div
          className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg cursor-pointer"
          onClick={toggleChat}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <span className="text-2xl">🤖</span>
        </motion.div>
        <AnimatePresence>
          {isChatOpen && (
            <motion.div
              className="fixed bottom-24 right-8 w-96 bg-white/10 backdrop-blur-sm rounded-lg shadow-2xl border border-gray-200"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ duration: 0.3 }}
            >
              <Chatbot apiKey="AIzaSyCsiDiFLSwhCvT-7pUX9SKVeL3VmWTInGg" modelType="gemini" />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default ClassificationPage;