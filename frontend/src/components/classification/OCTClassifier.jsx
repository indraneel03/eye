import React, { useState } from 'react';
import { Upload, RotateCcw } from 'lucide-react';
import GradCAMVisualization from './GradCAMVisualization';
import { classifyImage, getSHAPExplanation, getIntegratedGradients, getTCAVExplanation } from '../../utils/api';

const OCTClassifier = () => {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      // Classify the image
      const classificationResult = await classifyImage(file, 'mobilenet_v3');
      setResults(classificationResult);

      // Fetch additional explanations
      const shapResult = await getSHAPExplanation(file, 'mobilenet_v3');
      const igResult = await getIntegratedGradients(file, 'mobilenet_v3');
      const tcavResult = await getTCAVExplanation(file, 'mobilenet_v3');

      // Update results with additional explanations
      setResults((prev) => ({
        ...prev,
        shap: shapResult,
        ig: igResult,
        tcav: tcavResult,
      }));
    } catch (error) {
      console.error('Error processing image:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold text-center mb-8">
          OCT Image Classification
        </h1>

        {/* Upload Area */}
        <div className={`border-2 border-dashed border-blue-400 rounded-lg p-8 
          ${!results ? 'bg-gray-800/50' : 'bg-transparent'}`}>
          {!results ? (
            <div className="flex flex-col items-center justify-center">
              <Upload className="w-12 h-12 text-blue-400 mb-4" />
              <label htmlFor="image-upload" className="cursor-pointer">
                <span className="text-blue-400 text-xl">
                  Click or drag image to upload
                </span>
                <input
                  id="image-upload"
                  type="file"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageUpload}
                  disabled={loading}
                />
              </label>
            </div>
          ) : (
            <div className="space-y-8">
              {/* Results Section */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4">Results</h2>
                <div className="space-y-2">
                  <p className="text-lg">
                    <span className="font-medium">Condition:</span>{' '}
                    <span className="text-blue-400">{results.class}</span>
                  </p>
                  <p className="text-lg">
                    <span className="font-medium">Confidence:</span>{' '}
                    <span className="text-blue-400">
                      {results.confidence.toFixed(1)}%
                    </span>
                  </p>
                </div>
              </div>

              {/* GradCAM Visualization */}
              <GradCAMVisualization
                original={results.original}
                gradcam={results.gradcam}
                isLoading={loading}
              />

              {/* Add SHAP, IG, and TCAV visualizations here */}
            </div>
          )}
        </div>

        {/* Reset Button */}
        {results && (
          <div className="flex justify-center">
            <button
              onClick={handleReset}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg transition-colors"
            >
              <RotateCcw className="w-5 h-5" />
              Reset
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default OCTClassifier;