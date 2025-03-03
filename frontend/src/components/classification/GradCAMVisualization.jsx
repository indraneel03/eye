import React, { useState } from 'react';

const GradCAMVisualization = ({ original, gradcam, isLoading }) => {
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);

  return (
    <div className="mt-8 bg-white/10 backdrop-blur-sm rounded-lg p-6 shadow-2xl">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-blue-400">Grad-CAM Visualization</h3>
      </div>

      {isLoading ? (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-10 w-10 border-b-4 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-blue-400 font-medium">Loading Grad-CAM...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Original Image */}
          <div>
            <h4 className="text-md font-semibold text-blue-400 mb-2">Original Image</h4>
            <img
              src={original}
              alt="Original OCT"
              className="w-full h-48 object-cover rounded-lg"
            />
          </div>

          {/* Grad-CAM Heatmap */}
          <div>
            <h4 className="text-md font-semibold text-blue-400 mb-2">Grad-CAM Heatmap</h4>
            <img
              src={gradcam}
              alt="Grad-CAM Heatmap"
              className="w-full h-48 object-cover rounded-lg"
            />
          </div>

          {/* Overlay */}
          <div>
            <h4 className="text-md font-semibold text-blue-400 mb-2">Overlay</h4>
            <div className="relative h-48 w-full rounded-lg overflow-hidden">
              <img
                src={original}
                alt="Original OCT"
                className="w-full h-full object-cover"
              />
              <img
                src={gradcam}
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
        </div>
      )}
    </div>
  );
};

export default GradCAMVisualization;