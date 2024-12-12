import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  MenuItem,
  Paper,
  Typography,
  Grid,
} from '@mui/material';
import { useDispatch } from 'react-redux';
import { createKPI } from '../redux/slices/kpiSlice';

const timePeriods = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'quarterly', label: 'Quarterly' },
  { value: 'annually', label: 'Annually' },
];

export default function KPICreationForm() {
  const dispatch = useDispatch();
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    targetValue: '',
    minThreshold: '',
    maxThreshold: '',
    timePeriod: 'monthly',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    dispatch(createKPI({
      name: formData.name,
      description: formData.description,
      target_value: parseFloat(formData.targetValue),
      min_threshold: formData.minThreshold ? parseFloat(formData.minThreshold) : null,
      max_threshold: formData.maxThreshold ? parseFloat(formData.maxThreshold) : null,
      time_period: formData.timePeriod,
    }));
    // Reset form
    setFormData({
      name: '',
      description: '',
      targetValue: '',
      minThreshold: '',
      maxThreshold: '',
      timePeriod: 'monthly',
    });
  };

  return (
    <Paper elevation={3} sx={{ p: 3, maxWidth: 600, mx: 'auto', mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Create New KPI
      </Typography>
      <Box component="form" onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              required
              fullWidth
              label="KPI Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              multiline
              rows={3}
              value={formData.description}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              required
              fullWidth
              label="Target Value"
              name="targetValue"
              type="number"
              value={formData.targetValue}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              select
              label="Time Period"
              name="timePeriod"
              value={formData.timePeriod}
              onChange={handleChange}
            >
              {timePeriods.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Minimum Threshold"
              name="minThreshold"
              type="number"
              value={formData.minThreshold}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Maximum Threshold"
              name="maxThreshold"
              type="number"
              value={formData.maxThreshold}
              onChange={handleChange}
            />
          </Grid>
          <Grid item xs={12}>
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              size="large"
            >
              Create KPI
            </Button>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
}
