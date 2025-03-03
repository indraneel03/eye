const LoadingSpinner = () => (
  <div className="text-center">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
    <p className="mt-2">Processing image...</p>
  </div>
);

export default LoadingSpinner;