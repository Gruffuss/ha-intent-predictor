-- Migration 001: Initial schema setup
-- Creates all tables and hypertables for HA Intent Prediction System

\echo 'Running migration 001: Initial schema setup...'

-- Run the main schema
\i ../schema.sql

-- Insert initial configuration data
INSERT INTO model_configurations (room, model_name, version, hyperparameters, features_used, training_config, is_active) VALUES
('living_kitchen', 'adaptive_random_forest', '1.0', '{"n_models": 10, "max_features": "sqrt", "lambda_value": 6}', '{}', '{"online_learning": true}', true),
('bedroom', 'adaptive_random_forest', '1.0', '{"n_models": 10, "max_features": "sqrt", "lambda_value": 6}', '{}', '{"online_learning": true}', true),
('office', 'adaptive_random_forest', '1.0', '{"n_models": 10, "max_features": "sqrt", "lambda_value": 6}', '{}', '{"online_learning": true}', true),
('bathroom', 'adaptive_random_forest', '1.0', '{"n_models": 10, "max_features": "sqrt", "lambda_value": 6}', '{}', '{"online_learning": true}', true),
('small_bathroom', 'adaptive_random_forest', '1.0', '{"n_models": 10, "max_features": "sqrt", "lambda_value": 6}', '{}', '{"online_learning": true}', true);

\echo 'Migration 001 completed successfully!'