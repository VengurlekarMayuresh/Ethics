'use client';

import { useState } from 'react';
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { UploadCloud, CheckCircle2, ChevronRight, X, Plus, AlertCircle } from 'lucide-react';

const modelSchema = z.object({
  name: z.string().min(3, 'Name must be at least 3 characters'),
  description: z.string().optional(),
  framework: z.enum(['sklearn', 'xgboost', 'keras', 'onnx', 'api']),
  task_type: z.enum(['classification', 'regression']),
  features: z.array(z.object({
    name: z.string().min(1, 'Feature name is required'),
    type: z.enum(['numeric', 'categorical']),
    options: z.string().optional() // Comma separated for categorical
  })).min(1, 'At least one feature is required')
});

type ModelForm = z.infer<typeof modelSchema>;

export default function ModelUpload() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const { register, control, handleSubmit, watch, formState: { errors } } = useForm<ModelForm>({
    resolver: zodResolver(modelSchema),
    defaultValues: {
      framework: 'sklearn',
      task_type: 'classification',
      features: [{ name: '', type: 'numeric', options: '' }]
    }
  });

  const { fields, append, remove } = useFieldArray({
    control,
    name: 'features'
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const nextStep = () => {
    if (step === 1 && !file) {
      setError('Please upload a model file');
      return;
    }
    setError('');
    setStep(s => s + 1);
  };

  const prevStep = () => {
    setStep(s => s - 1);
  };

  const onSubmit = async (data: ModelForm) => {
    if (!file) {
      setError('Model file is required');
      return;
    }

    try {
      setIsSubmitting(true);
      setError('');
      
      const formattedFeatures = data.features.map(f => ({
        name: f.name,
        type: f.type,
        options: f.type === 'categorical' && f.options ? f.options.split(',').map(s => s.trim()) : []
      }));

      const formData = new FormData();
      formData.append('name', data.name);
      formData.append('description', data.description || '');
      formData.append('framework', data.framework);
      formData.append('task_type', data.task_type);
      formData.append('feature_schema', JSON.stringify(formattedFeatures));
      formData.append('file', file);

      await api.post('/models/upload', formData);

      router.push('/models');
    } catch (err: any) {
      console.error(err);
      const detail = err.response?.data?.detail;
      const errorMessage = typeof detail === 'string' 
        ? detail 
        : (Array.isArray(detail) ? JSON.stringify(detail) : 'Failed to upload model. Please check the API server.');
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-8">
      {/* Progress Bar */}
      <nav aria-label="Progress">
        <ol role="list" className="flex items-center">
          <li className={`relative pr-8 sm:pr-20 ${step >= 1 ? 'text-indigo-600' : 'text-gray-400'}`}>
            <div className="flex items-center">
              <span className={`flex h-8 w-8 items-center justify-center rounded-full ${step >= 1 ? 'bg-indigo-600 text-white' : 'border-2 border-gray-300'}`}>
                {step > 1 ? <CheckCircle2 className="h-5 w-5" /> : '1'}
              </span>
              <span className="ml-4 text-sm font-medium">Model Details</span>
            </div>
          </li>
          <li className={`relative pr-8 sm:pr-20 ${step >= 2 ? 'text-indigo-600' : 'text-gray-400'}`}>
             <div className="flex items-center">
              <span className={`flex h-8 w-8 items-center justify-center rounded-full ${step >= 2 ? 'bg-indigo-600 text-white' : 'border-2 border-gray-300'}`}>
                {step > 2 ? <CheckCircle2 className="h-5 w-5" /> : '2'}
              </span>
              <span className="ml-4 text-sm font-medium">Input Features</span>
            </div>
          </li>
        </ol>
      </nav>

      {error && (
        <div className="mt-8 rounded-md bg-red-50 p-4 border border-red-200 flex items-start">
          <AlertCircle className="h-5 w-5 text-red-400 mt-0.5 mr-3 flex-shrink-0" />
          <div className="text-sm text-red-700">{error}</div>
        </div>
      )}

      {/* Form Steps */}
      <div className="mt-8 rounded-xl bg-white shadow-sm border border-gray-200">
        <form onSubmit={handleSubmit(onSubmit)}>
          
          {/* Step 1: Basic Info & File Upload */}
          <div className={`p-8 ${step !== 1 ? 'hidden' : ''}`}>
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Step 1: Upload & Initial Metadata</h2>
            
            <div className="space-y-6">
              {/* File Upload Area */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Model File</label>
                <div className="mt-1 flex justify-center rounded-xl border-2 border-dashed border-gray-300 px-6 pt-5 pb-6 hover:bg-gray-50 transition-colors">
                  <div className="space-y-1 text-center">
                    <UploadCloud className="mx-auto h-12 w-12 text-indigo-500" />
                    <div className="flex text-sm text-gray-600 mt-4 justify-center">
                      <label htmlFor="file-upload" className="relative cursor-pointer rounded-md bg-white font-medium text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 hover:text-indigo-500">
                        <span>Upload a file</span>
                        <input id="file-upload" type="file" className="sr-only" onChange={handleFileChange} accept=".pkl,.joblib,.h5,.onnx,.json" />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">.pkl, .joblib, .onnx, .h5, .json up to 100MB</p>
                    {file && (
                      <div className="mt-4 p-2 bg-indigo-50 rounded text-indigo-700 text-sm font-medium flex items-center justify-center">
                        <CheckCircle2 className="h-4 w-4 mr-2" />
                        Selected: {file.name} ({(file.size / (1024*1024)).toFixed(2)} MB)
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700">Model Name</label>
                  <input type="text" {...register('name')} className="mt-1 block w-full rounded-md border border-gray-300 py-2 px-3 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm" />
                  {errors.name && <p className="mt-1 text-xs text-red-500">{errors.name.message}</p>}
                </div>
                
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700">Description (Optional)</label>
                  <textarea {...register('description')} rows={3} className="mt-1 block w-full rounded-md border border-gray-300 py-2 px-3 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm" />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Framework</label>
                  <select {...register('framework')} className="mt-1 block w-full rounded-md border border-gray-300 py-2 px-3 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm">
                    <option value="sklearn">Scikit-Learn</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="keras">Keras (TensorFlow)</option>
                    <option value="onnx">ONNX Runtime</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Task Type</label>
                  <select {...register('task_type')} className="mt-1 block w-full rounded-md border border-gray-300 py-2 px-3 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:text-sm">
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-end pt-4">
                <button type="button" onClick={nextStep} className="inline-flex items-center rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none">
                  Next Step
                  <ChevronRight className="ml-2 h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Step 2: Feature Schema */}
          <div className={`p-8 ${step !== 2 ? 'hidden' : ''}`}>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Step 2: Model Input Features</h2>
            <p className="text-gray-500 text-sm mb-6">Define the exact inputs your model expects. This will be used to generate prediction forms and label SHAP charts.</p>
            
            <div className="space-y-4">
              {fields.map((field, index) => {
                const type = watch(`features.${index}.type`);
                return (
                  <div key={field.id} className="p-4 border border-gray-200 rounded-lg bg-gray-50 relative">
                    {index > 0 && (
                      <button type="button" onClick={() => remove(index)} className="absolute top-2 right-2 text-gray-400 hover:text-red-500">
                        <X className="h-5 w-5" />
                      </button>
                    )}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">Feature Name</label>
                        <input {...register(`features.${index}.name`)} className="block w-full rounded-md border border-gray-300 py-1.5 px-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" placeholder="e.g. age, income" />
                        {errors.features?.[index]?.name && <p className="mt-1 text-xs text-red-500">{errors.features[index].name.message}</p>}
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">Data Type</label>
                        <select {...register(`features.${index}.type`)} className="block w-full rounded-md border border-gray-300 py-1.5 px-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500">
                          <option value="numeric">Numeric (Float/Int)</option>
                          <option value="categorical">Categorical</option>
                        </select>
                      </div>
                      
                      {type === 'categorical' && (
                        <div className="md:col-span-2">
                          <label className="block text-xs font-medium text-gray-700 mb-1">Categories (Comma separated)</label>
                          <input {...register(`features.${index}.options`)} className="block w-full rounded-md border border-gray-300 py-1.5 px-3 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" placeholder="Low, Medium, High" />
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <button type="button" onClick={() => append({ name: '', type: 'numeric', options: '' })} className="mt-4 flex items-center text-sm font-medium text-indigo-600 hover:text-indigo-700">
              <Plus className="mr-1 h-4 w-4" /> Add Feature
            </button>

            <div className="flex justify-between pt-8 border-t border-gray-100 mt-8">
              <button type="button" onClick={prevStep} className="inline-flex items-center rounded-lg bg-white border border-gray-300 px-6 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none">
                Back
              </button>
              <button type="submit" disabled={isSubmitting} className="inline-flex items-center rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none disabled:bg-indigo-400">
                {isSubmitting ? 'Uploading...' : 'Deploy Model'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
