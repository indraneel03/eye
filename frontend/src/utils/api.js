import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

// Classify an image using the selected model
export const classifyImage = async (file, model = 'mobilenet_v3') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    const response = await axios.post(`${API_BASE_URL}/classify`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Classification error:', error);
    throw new Error(error.response?.data?.error || 'Failed to classify image');
  }
};

// Fetch SHAP explanation for the uploaded image
export const getSHAPExplanation = async (file, model = 'mobilenet_v3') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    const response = await axios.post(`${API_BASE_URL}/shap`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('SHAP explanation error:', error);
    throw new Error(error.response?.data?.error || 'Failed to generate SHAP explanation');
  }
};

// Fetch Integrated Gradients explanation for the uploaded image
export const getIntegratedGradients = async (file, model = 'mobilenet_v3') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    const response = await axios.post(`${API_BASE_URL}/integrated-gradients`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Integrated Gradients error:', error);
    throw new Error(error.response?.data?.error || 'Failed to generate Integrated Gradients');
  }
};

// Fetch TCAV explanation for the uploaded image
export const getTCAVExplanation = async (file, model = 'mobilenet_v3') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);

    const response = await axios.post(`${API_BASE_URL}/tcav`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('TCAV explanation error:', error);
    throw new Error(error.response?.data?.error || 'Failed to generate TCAV explanation');
  }
};