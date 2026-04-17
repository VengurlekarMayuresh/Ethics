import axios from 'axios';
import { useStore } from './store';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_URL.endsWith('/') ? API_URL : `${API_URL}/`,
});

api.interceptors.request.use(
  (config) => {
    // Strip leading slash if present to ensure the baseURL prefix is respected
    if (config.url?.startsWith('/')) {
      config.url = config.url.substring(1);
    }
    const token = useStore.getState().token;
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);
