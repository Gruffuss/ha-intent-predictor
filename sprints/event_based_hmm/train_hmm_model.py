#!/usr/bin/env python3
"""
Sprint 3: Train HMM on bedroom transition sequences
Implement basic HMM using hmmlearn on real bedroom occupancy data
"""

from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader
import asyncio
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
import json

# Import HMM libraries
try:
    from hmmlearn import hmm
    import sklearn
except ImportError:
    print("Warning: hmmlearn not installed. Will show installation command.")

async def load_transition_sequences() -> Tuple[List[List[str]], List[List[str]]]:
    """Load bedroom transition sequences from database"""
    print("=== LOADING BEDROOM TRANSITION SEQUENCES ===")
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    await db.initialize()
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Get ALL states in chronological order and compute transitions
        result = await conn.execute(text("""
            SELECT 
                state,
                timestamp
            FROM sensor_events 
            WHERE entity_id = 'binary_sensor.bedroom_presence_sensor_full_bedroom'
            ORDER BY timestamp ASC
        """))
        all_events = result.fetchall()
        
        # Compute transitions manually
        transitions = []
        if len(all_events) > 1:
            prev_state = all_events[0][0]
            for current_state, timestamp in all_events[1:]:
                if current_state != prev_state:
                    transitions.append((current_state, timestamp))
                    prev_state = current_state
    
    await db.close()
    
    # Convert to state sequences
    states = [state for state, _ in transitions]
    sequence_length = 8
    
    # Create sequences
    sequences = []
    for i in range(len(states) - sequence_length + 1):
        sequence = states[i:i + sequence_length]
        sequences.append(sequence)
    
    # Split train/test
    split_point = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_point]
    test_sequences = sequences[split_point:]
    
    print(f"Loaded {len(transitions):,} transitions")
    print(f"Created {len(sequences):,} sequences")
    print(f"Training: {len(train_sequences):,}, Testing: {len(test_sequences):,}")
    
    return train_sequences, test_sequences

def encode_sequences(sequences: List[List[str]]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode string sequences to numerical arrays for HMM training
    
    Returns:
        observations: Numerical sequences
        state_to_num: Mapping from state names to numbers
        num_to_state: Mapping from numbers to state names
    """
    print("=== ENCODING SEQUENCES FOR HMM ===")
    
    # Get all unique states
    all_states = set()
    for seq in sequences:
        all_states.update(seq)
    
    # Create mappings
    state_to_num = {state: i for i, state in enumerate(sorted(all_states))}
    num_to_state = {i: state for state, i in state_to_num.items()}
    
    print(f"Unique states found: {sorted(all_states)}")
    print(f"State mappings: {state_to_num}")
    
    # Convert sequences to numerical arrays
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [state_to_num[state] for state in seq]
        encoded_sequences.append(encoded_seq)
    
    # Convert to numpy array format for hmmlearn
    observations = np.array(encoded_sequences)
    
    print(f"Encoded {len(encoded_sequences):,} sequences")
    print(f"Observation shape: {observations.shape}")
    
    return observations, state_to_num, num_to_state

def create_hmm_model(n_hidden_states: int = 4) -> object:
    """
    Create HMM model with specified hidden states
    
    Hidden states represent occupancy patterns:
    - State 0: Away (long periods unoccupied)  
    - State 1: Arriving (transitioning to occupied)
    - State 2: Occupied (actively in bedroom)
    - State 3: Leaving (transitioning to unoccupied)
    """
    print(f"=== CREATING HMM MODEL ===")
    print(f"Hidden states: {n_hidden_states}")
    print("State meanings:")
    print("  State 0: Away (extended unoccupied)")
    print("  State 1: Arriving (entering bedroom)")  
    print("  State 2: Occupied (actively present)")
    print("  State 3: Leaving (exiting bedroom)")
    
    # Create Gaussian HMM model
    model = hmm.GaussianHMM(
        n_components=n_hidden_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    return model

def train_hmm_model(model: object, train_observations: np.ndarray) -> object:
    """Train HMM on bedroom transition sequences"""
    print("=== TRAINING HMM MODEL ===")
    print(f"Training data shape: {train_observations.shape}")
    
    # Reshape for hmmlearn (needs 2D: [n_samples, n_features])
    # Each sequence becomes a separate sample
    X_train = train_observations.reshape(-1, 1).astype(np.float64)
    lengths = [train_observations.shape[1]] * train_observations.shape[0]
    
    print(f"Reshaped training data: {X_train.shape}")
    print(f"Sequence lengths: {len(lengths)} sequences of length {lengths[0] if lengths else 0}")
    
    try:
        # Train the model
        model.fit(X_train, lengths)
        print("✅ HMM training completed successfully!")
        
        # Display learned parameters
        print(f"\nLearned transition matrix shape: {model.transmat_.shape}")
        print(f"Learned emission parameters shape: {model.means_.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ HMM training failed: {e}")
        return None

def analyze_hmm_parameters(model: object, num_to_state: Dict[int, str]) -> None:
    """Analyze learned HMM parameters"""
    print("=== ANALYZING LEARNED HMM PARAMETERS ===")
    
    if model is None:
        print("No trained model to analyze")
        return
    
    # Transition matrix
    print("\nTransition Matrix (hidden state transitions):")
    print("Rows: current state, Columns: next state")
    transition_matrix = model.transmat_
    
    state_names = ["Away", "Arriving", "Occupied", "Leaving"]
    print(f"{'':>10} {'Away':>8} {'Arriving':>8} {'Occupied':>8} {'Leaving':>8}")
    
    for i, current_state in enumerate(state_names):
        row = f"{current_state:>10}"
        for j in range(transition_matrix.shape[1]):
            prob = transition_matrix[i, j]
            row += f" {prob:>7.3f}"
        print(row)
    
    # Emission parameters (means)
    print(f"\nEmission Means (what each hidden state 'sees'):")
    for i, state_name in enumerate(state_names):
        mean_val = model.means_[i, 0]
        print(f"  {state_name}: {mean_val:.3f}")
    
    # Most likely transitions
    print(f"\nMost likely transitions:")
    for i, current_state in enumerate(state_names):
        max_j = np.argmax(transition_matrix[i, :])
        max_prob = transition_matrix[i, max_j]
        next_state = state_names[max_j]
        print(f"  {current_state} → {next_state} ({max_prob:.3f})")

def test_hmm_prediction(model: object, test_sequences: List[List[str]], 
                       state_to_num: Dict[str, int], num_to_state: Dict[int, str]) -> None:
    """Test HMM prediction capability"""
    print("=== TESTING HMM PREDICTION ===")
    
    if model is None:
        print("No trained model for testing")
        return
    
    # Test on first few sequences
    for i, test_seq in enumerate(test_sequences[:3]):
        print(f"\nTest sequence {i+1}: {' → '.join(test_seq)}")
        
        # Encode sequence
        encoded_seq = np.array([[state_to_num[state]] for state in test_seq]).astype(np.float64)
        
        try:
            # Predict most likely hidden state sequence
            logprob, hidden_states = model.decode(encoded_seq, algorithm="viterbi")
            
            state_names = ["Away", "Arriving", "Occupied", "Leaving"]
            hidden_sequence = [state_names[state] for state in hidden_states]
            
            print(f"Hidden states: {' → '.join(hidden_sequence)}")
            print(f"Log probability: {logprob:.3f}")
            
        except Exception as e:
            print(f"Prediction failed: {e}")

async def main():
    """Main function for Sprint 3: HMM Implementation"""
    print("SPRINT 3: Event-Based HMM Training")
    print("=" * 50)
    
    try:
        # Check if hmmlearn is available
        import hmmlearn
        print("✅ hmmlearn library available")
    except ImportError:
        print("❌ hmmlearn not installed")
        print("Install with: pip install hmmlearn")
        return
    
    # Step 1: Load transition sequences
    print("\nStep 1: Loading transition sequences...")
    train_sequences, test_sequences = await load_transition_sequences()
    
    # Step 2: Encode sequences for HMM
    print("\nStep 2: Encoding sequences...")
    train_obs, state_to_num, num_to_state = encode_sequences(train_sequences)
    
    # Step 3: Create HMM model
    print("\nStep 3: Creating HMM model...")
    model = create_hmm_model(n_hidden_states=4)
    
    # Step 4: Train HMM
    print("\nStep 4: Training HMM...")
    trained_model = train_hmm_model(model, train_obs)
    
    # Step 5: Analyze learned parameters
    print("\nStep 5: Analyzing HMM parameters...")
    analyze_hmm_parameters(trained_model, num_to_state)
    
    # Step 6: Test predictions
    print("\nStep 6: Testing HMM predictions...")
    test_hmm_prediction(trained_model, test_sequences, state_to_num, num_to_state)
    
    # Summary
    print(f"\n=== SPRINT 3 DELIVERABLE SUMMARY ===")
    if trained_model is not None:
        print(f"✅ HMM model trained successfully")
        print(f"✅ 4 hidden states: Away, Arriving, Occupied, Leaving")
        print(f"✅ Learned transition probabilities")
        print(f"✅ Model ready for prediction testing")
        print(f"✅ Ready for Sprint 4: Transition Prediction")
    else:
        print(f"❌ HMM training failed - check dependencies")
    
    return trained_model

if __name__ == "__main__":
    result = asyncio.run(main())