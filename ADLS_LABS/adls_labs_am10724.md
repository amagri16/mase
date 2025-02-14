# Advanced Deep Learning Systems: Lab Report

---

# Lab 1: Model Compression (Quantization and Pruning)

## Implementation Tasks

### Task 1: Quantization with Varying Fixed-Point Widths

1. **Quantize the Model:** every Linear layer in the model is quantized using a configuration that sets the fixed-point width. We explore widths from **4 to 32 bits**.
2. **Evaluation and Plotting:** for each fixed-point width, we record:

   - **PTQ (Post-Training Quantization) Accuracy**
   - **QAT (Quantization-Aware Training) Accuracy**

   A plot is generated with:

   - **x-axis:** Fixed-point width (bits)
   - **y-axis:** Accuracy on the IMDb dataset

   Separate curves for PTQ and QAT illustrate the impact of post-quantization fine-tuning.


<img src="lab1/qatvsptq.png" alt="Figure 1: IMDb Accuracy vs. Quantization Precision" width="800">

Quantization results show a substantial jump in accuracy between 4-bit and 8-bit precision for both PTQ and QAT, with performance improvements diminishing after 8 bits. We can also see that QAT consistently outperforms PTQ across all tested bit widths, demonstrating the benefits of fine-tuning under quantization constraints. The best trade-off is then at 8 bits, where near-floating-point accuracy is achieved while significantly reducing model size, compared to higher widths.

## Task 2: Pruning with Varying Sparsity

After obtaining the best performing quantized model from Task 1 (in this case, the 8-bit precision model), we apply model pruning to further reduce model complexity while maintaining accuracy. In this task, we explore two pruning strategies: **Random Pruning** and **L1-Norm Pruning**. The sparsity parameter is varied from **0.1** to **0.9**.

### Procedure

1. **Load Best Model:**  
   Start with the best model from Task 1.

2. **Vary Sparsity:**  
   For each sparsity level:
   - **Random Pruning:** randomly remove the specified fraction of weights.
   - **L1-Norm Pruning:** remove weights based on their L1-norm, targeting the less important ones.

3. **Evaluation:**  
   After pruning, each model is evaluated on the IMDb dataset to measure accuracy.

4. **Plotting:**  
   A plot is generated with:
   - **x-axis:** Sparsity level
   - **y-axis:** Accuracy on the IMDb dataset

**Relevant Code for Task 2:**
```python
# ----------------------------------------------------------------
# 3. Run pruning experiments varying sparsity and comparing methods
# ----------------------------------------------------------------
pruning_methods = ["random", "l1-norm"]
sparsity_values = [0.1 * i for i in range(1, 10)]  # 0.1, 0.2, …, 0.9

# Dictionary to record final accuracy for each method
results = {method: [] for method in pruning_methods}

for method in pruning_methods:
    print(f"\n=== Pruning Method: {method} ===")
    for sparsity in sparsity_values:
        print(f"\n--- Testing sparsity: {sparsity:.1f} ---")
        
        # Create a fresh copy of the best QAT model for each experiment
        pruned_mg = copy.deepcopy(best_mg)
        
        # Define the pruning configuration for both weights and activations
        pruning_config = {
            "weight": {
                "sparsity": sparsity,
                "method": method,
                "scope": "local",
            },
            "activation": {
                "sparsity": sparsity,
                "method": method,
                "scope": "local",
            },
        }
        
```
Separate curves illustrate the performance differences between **Random** and **L1-Norm** pruning.


<img src="lab1/pruning.png" alt="Figure 2: Final accuracy after pruning and fine tuning" width="800">


### Summary of Results

- **Low Sparsity (0.1–0.3):** both Random and L1-Norm pruning maintain high accuracy (~86–87%).
- **Moderate Sparsity (0.4–0.5):** L1-Norm pruning outperforms Random pruning, preserving accuracy around 87%, while Random pruning starts to decline.
- **High Sparsity (0.6–0.9):** both methods see a significant drop in accuracy, with L1-Norm pruning gradually degrading until a sharp decline at very high sparsity.

Overall, L1-Norm pruning is more effective at retaining performance at moderate sparsity levels compared to Random pruning.

---

# Lab 2: Neural Architecture Search


## Implementation Tasks

### Task 1. Hyperparameter and Architecture Search

- **Baseline:**  
  Tutorial 5 demonstrates using random search to find the optimal configuration of hyperparameters and layer choices for the BERT model.
  
- **Extended Exploration:**  
  Now, extend the search by using:
  - **GridSampler**
  - **TPESampler**

- **Evaluation:**  
  Plot a figure with:
  - **x-axis:** Number of trials
  - **y-axis:** Maximum achieved accuracy up to that point

  Include one curve for each sampler (GridSampler and TPESampler) to compare their performance.

<img src="lab2/task1.png" alt="Figure 3: Final accuracy after pruning and fine tuning" width="800">


### Summary of Results

- **Sampler Comparison:**  
  - **TPESampler:**  TPE outperforms both Grid and Random search, likely because it balances exploration and exploitation, adapting based on previous trials.
  - **GridSampler:**  the GridSampler curve shows a more gradual improvement, reflecting its systematic exploration of the parameter grid. While it may take longer to reach peak accuracy, it provides a thorough search that can be beneficial when the hyperparameter space is well understood.
  - **Random Sampler:** Random search got very lucky at the first trial and struggles to improve accuracy, highlighting why more informed search strategies are valuable.

---

### Task 2. Compression-Aware NAS

- **Motivation:**  
  In Tutorial 5, after finding an optimal configuration via NAS, the CompressionPipeline in Mase is used to quantize and prune the model. However, different model architectures may have varying sensitivities to compression, meaning that the best architecture found without compression might not be optimal once quantized and pruned.

- **Objective:**  
  Develop a compression-aware search flow where quantization and pruning are considered within each trial.

- **Method:**
  - In the objective function, after constructing and initially training the model for a few iterations, invoke the CompressionPipeline to quantize and prune the model.
  - Use the sampler that yielded the best results in Task 1 for the compression-aware search.
  - The objective function should return the final accuracy of the model after compression.
  - Consider both scenarios:
    1. **Without post-compression training**
    2. **With post-compression training**

- **Evaluation:**  
  Plot a new figure with:
  - **x-axis:** Number of trials
  - **y-axis:** Maximum achieved accuracy up to that point
  
  This plot should display three curves:
  1. Best performance from Task 1 (without compression)
  2. Compression-aware search without post-compression training
  3. Compression-aware search with post-compression training


**Relevant Code for Task 2:**

```python
    # 2) Apply Compression (quantization and pruning)
    mg = MaseGraph(model)
    pipe = CompressionPipeline()
    mg, _ = pipe(mg, pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    })
    
    # 3) Evaluate without additional training
    mg.model.to("cuda")
    wrapped_model = HuggingFaceCompatibleModel(mg.model).to("cuda")
    trainer_eval = get_trainer(
        model=wrapped_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0,
    )
    eval_results = trainer_eval.evaluate()
    final_acc = eval_results["eval_accuracy"]
    trial.set_user_attr("model", wrapped_model)
    return final_acc


    
    # 2) Apply Compression (quantization and pruning)
    mg = MaseGraph(model)
    pipe = CompressionPipeline()
    mg, _ = pipe(mg, pass_args={
        "quantize_transform_pass": quantization_config,
        "prune_transform_pass": pruning_config,
    })
    mg.model.to("cuda")
    
    # 3) Post-compression training
    wrapped_model = HuggingFaceCompatibleModel(mg.model).to("cuda")
    trainer_post = get_trainer(
        model=wrapped_model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer_post.train()
```

<img src="lab2/task2.png" alt="Figure 4: Final accuracy after pruning and fine tuning" width="800">
### Summary of Results

**1. Baseline (No Compression):**

This represents the best performance from Task 1, where the model is optimized without considering quantization or pruning. The accuracy remains consistently high (~0.85) across trials, suggesting that the best architecture found in NAS is robust when left uncompressed.

**2. Compression-Aware Search Without Post-Training:**

Initially, this performs significantly worse than the baseline, likely because pruning and quantization degrade model performance without further fine-tuning. It improves steadily, reaching around 0.75 accuracy but plateaus after a few trials. This suggests that quantization and pruning introduce substantial degradation that NAS alone cannot fully mitigate.

 **3. Compression-Aware Search With Post-Training:**

This approach performs the best among compression-aware methods, it closely matches the baseline's accuracy (~0.85), indicating that fine-tuning after compression helps recover lost performance. There is a slight gap between this and the baseline, which could suggest that compression still imposes some constraints on the optimal architecture.


---


## Lab 3: Mixed Precision Search


### Implementation Tasks

### Task 1: Layer-Specific Precision Allocation

#### **Objective**
- Modify the existing NAS setup to allow each `IntegerLinear` layer to have independent precision settings.
- The following options should be exposed as hyperparameters in Optuna:
  - **Width**: `{8, 16, 32}`
  - **Fractional Width**: `{2, 4, 8}`

#### **Steps**
1. Modify the code to allow **layer-wise precision allocation** instead of using a single global width.
2. Expose the precision choices as additional hyperparameters for **Optuna**.
3. Run the NAS search again.
4. **Plot the results**:
   - **x-axis**: Number of trials
   - **y-axis**: Maximum achieved accuracy up to that point.


**Relevant Code for Task 1:**
```python
# ---------------------------------------------------
# 3. Modified Model Constructor with New Hyperparams
# ---------------------------------------------------
def construct_model(trial):
    # Start with a fresh copy of the base model.
    trial_model = deepcopy(base_model)

    # Loop over all sub-modules in the model.
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            # Decide which linear layer type to use for this layer.
            new_layer_cls = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            # If the chosen type is the standard Linear, do nothing.
            if new_layer_cls == torch.nn.Linear:
                continue

            # Common arguments for any linear layer.
            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # If we choose the low-precision integer layer, sample additional hyperparameters.
            if new_layer_cls == LinearInteger:
                config = {
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_frac_width": trial.suggest_categorical(f"{name}_data_in_frac_width", [2, 4, 8]),
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_frac_width": trial.suggest_categorical(f"{name}_weight_frac_width", [2, 4, 8]),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_frac_width": trial.suggest_categorical(f"{name}_bias_frac_width", [2, 4, 8]),
                }
                kwargs["config"] = config

            # Create the new layer (copying over the weights).
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data

            # Replace the original layer in the model.
            deepsetattr(trial_model, name, new_layer)

    return trial_model
```

<img src="lab3/task1.png" alt="Figure 5" width="800">

---

### Task 2: Expanding Supported Precision Types

#### **Objective**
- Extend the search space to include **all supported precisions** for `Linear` layers in **Mase**, such as:
  - `nn.Linear` (Full precision baseline)
  - `IntegerLinear`
  - `MinifloatDenorm,`
  - `MinifloatIEEE`
  - `LinearLog`
  - `BlockFP`
  - `BlockLog`
  - `Binary`
  - `BinaryScaling`

#### **Steps**
1. Modify the search space to incorporate **all precision types**.
2. Update the model constructor to properly handle different layer types and their required arguments.
3. Re-run the NAS search with the extended precision search space.
4. **Plot the results**:
   - **x-axis**: Number of trials
   - **y-axis**: Maximum achieved accuracy up to that point.
   - **Multiple curves**: One curve for each precision type to compare performance.


<img src="lab3/task2.png" alt="Figure 6" width="800">

Zoomed Plot for more emphasis on some precision types:

<img src="lab3/task2_zoomed.png" alt="Figure 7" width="800">


#### Precision Types and Model Performance**

##### **High-Performing Precision Types (Best Accuracy, Fast Convergence)**
- **Linear (Full Precision)**
  Achieves the best accuracy consistently, highlighting that maintaining floating-point precision offers the least resistance to training. Expected, as full precision retains the highest information fidelity and avoids numerical errors introduced by quantization.

- **LinearMiniFloatIEEE**
  Slightly below full precision but still converges rapidly, suggesting that IEEE-standardized mini-float formats are well-optimized for deep learning. Indicates that reducing bit-width while keeping IEEE-like exponent management preserves accuracy.

- **LinearMinifloatDenorm**
  Close to IEEE minifloat but slightly lower, possibly due to denormalization handling. This suggests that while denormalized numbers help with extreme values, they may introduce minor instability compared to strict IEEE-compliant floats.

##### **Mid-Tier Precision Types (Acceptable Accuracy, Slower Convergence)**
- **LinearInteger**
  Starts lower but improves with trials, showing that integer-based quantization still works well but requires tuning.Likely struggles at initialization due to loss of floating-point dynamic range but benefits from later adjustments.

- **LinearBlockFP**
  Similar behavior to integer quantization—slower initial accuracy but catches up, indicating that block-based floating point may need optimization for weight distributions. The block format could introduce quantization noise that only stabilizes after multiple trials.

- **LinearBinaryScaling**
  - Initially performs worse but improves over time. Since binary activations and weights restrict information flow, more trials may be needed to reach decent performance.

##### **Low-Performing Precision Types (Significant Accuracy Drop)**
- **LinearBinary**
  Stagnates well below full precision models, reinforcing that raw binary representations alone significantly limit learning. The steep loss in accuracy is expected due to extreme quantization—without careful optimization, these models lose too much information.

- **LinearLog (Gray) & LinearBlockLog**
  - These perform the worst, stuck near 50% accuracy, indicating that logarithmic quantization struggles with this task.
  Logarithmic quantization compresses large values efficiently but is not well-suited for the balanced weight distributions in deep networks.

---

# Lab 4 (Software Stream) - Performance Engineering


## Implementation Tasks

### **1. Torch Compile Optimization**
- In the first part of Lab 4 (**torch.compile**), we did not observe significant run-time speedups.
- **Modifications & Investigation:**
  - Modify the code and investigate why this is the case.

```python
# [Keep the helper functions timed_gpu, timed_cpu, get_data, time_model unchanged]
# Define device
device = "cpu"
n = 100  # Increased number of runs to amortize compilation overhead


# Move model to target device and set to eval mode
model = model.to(device)
model.eval()

(...)

Original model: 8.3620 s
Optimized model: 5.7346 s
```
Set model to Evaluation Mode: the model might be in training mode, causing batch normalization layers to update their running statistics with each forward pass. This dynamic behavior triggers recompilation every time the model is run. Switching to evaluation mode stabilizes the model's state

Increase the number of runs: the initial compilation overhead is amortized over more runs, making the average time more reflective of the optimized performance. In addition, the performance of torch.compile improves over multiple runs because compiled kernels are cached.

  - If you change the device to **CUDA**, do you observe the same thing?
```python
Original model: 0.1132 s
Optimized model: 0.1128 s
```
Changing the device to cuda in the original code ( n = 5 ) actually has a huge impact on the run-time speedups. There is a small improvement in the optimized model, but not very significant. That might be explained by the fact that cuda backend in PyTorch is already highly optimized, so the benefits of the just-in-time compilation are much less noticeable on GPU compared to CPU. Additionally, when measuring on cuda, factors such as asynchronous execution and warm-up effects can mask small differences. Essentially, while torch.compile can lead to significant improvements on CPU by reducing python overhead and fusing operations, its impact on cuda is minimal because the GPU operations are already efficient. 

---

### **2. Kernel Fusion and SDPA Profiling**
- In the second part of Lab 4 (**kernel fusion**), we analyzed a **fused SDPA kernel**.
- **Extended Profiling:**
  - Extend the profiling to the **SDPA kernel**.
  - Compare its **runtime behavior** with the naive implementation.
  - If you change the device to **CUDA**, do you observe the same thing?

**CPU:**
```python
Naive SDPA Time (CPU): 0.714111 s
Fused SDPA Time (CPU): 0.028388 s
```

**CUDA:**
```python
Naive SDPA Time (CUDA): 392.860 μs
Fused SDPA Time (CUDA): 206.846 μs
```
**Detailed profiling using torch.profiler to analyze the execution behavior:**

**Naive SDPA (CPU)**

- Total CPU time: 0.714111 s

The naive implementation performs the SDPA computation as separate operations (matrix multiplication, softmax, and a second matrix multiplication).
This leads to multiple function calls and intermediate memory transfers, resulting in significant overhead on the CPU.

**Fused SDPA (CPU)**

- Total CPU time: 0.028388 s

In contrast, the fused implementation combines all the steps of the SDPA 
computation into a single operation. This eliminates the overhead of 
multiple function calls and reduces unnecessary memory traffic.


The fused SDPA kernel dramatically outperforms the naive version on CPU,
with the fused approach being roughly 25x faster. This performance boost 
is due to the reduction in dispatch overhead and the elimination of 
redundant memory operations.

**Naive SDPA (CUDA)**

- Total CUDA time: 392.860 μs

Key operations observed:
  - aten::matmul:    225.182 μs (57.32% of total CUDA time)
  - aten::softmax:    99.135 μs (25.23% of total CUDA time)
  - aten::mul:         68.543 μs (17.45% of total CUDA time)

Multiple kernel launches:
Each operation (matrix multiplication, softmax, and elementwise multiplication)
is executed in its own kernel. This separation leads to extra overhead due to 
intermediate memory transfers between global memory and faster on-chip caches.

**Fused SDPA (CUDA)**

- Total CUDA time: 206.846 μs (almost 50% faster than Naive SDPA)
Key operation:
  - aten::scaled_dot_product_attention: 206.846 μs (accounts for 100% of CUDA time)

Single kernel launch:

The fused kernel consolidates the entire attention computation 
(matmul → softmax → matmul) into one highly optimized kernel. By eliminating 
intermediate memory read/writes and reducing kernel launch overhead, it 
achieves significant performance gains.


---

### **3. Custom Kernel Development (MXINT8 Dequantization)**
- In the third part of Lab 4 (**Custom kernel**), we explored **MXINT8 dequantization** and its integration with Python.

#### **Questions to Address:**
**a)** How does **MXINT8** benefit custom hardware if both the activation and weights in a linear layer are quantized to **MXINT8**?  

**Answer:**
When both activations and weights are stored in MXINT8 form, custom hardware can exploit that common exponent and the compact mantissa storage to achieve faster, lower-memory multiply-accumulate operations, for the following reasons:

- **Reduced bandwidth and memory footprint**

Storing parameters in 8 bits (rather than 32) cuts the memory needed for weights and activations.
Smaller data transfers mean the hardware’s memory interface can serve more values per clock, boosting throughput.


- **Shared exponent for a group of mantissas**

Unlike a pure fixed-point format or float (which has a separate exponent per value), MXINT groups mantissas that share one exponent.
This grouping preserves a dynamic range closer to floating point while staying compact like fixed-point, and so, on custom hardware, it’s simpler to apply one exponent shift for an entire block of data rather than decoding each value’s exponent independently. That means fewer exponent-handling circuits and fewer per-value overheads.

- **Efficient multiply-accumulate**

With MXINT8, both the weight and activation multipliers are 8-bit mantissas. A single 8-bit multiplier can handle each mantissa multiplication directly.
The hardware only needs to adjust for the single shared exponent once per group rather than doing exponent manipulations for every multiply.


**b)** What is the purpose of the variables **`dont_need_abs`** and **`bias`** in the C++ for loop? 

**Answer:**
`dont_need_abs` and `bias` together handle the fact that MXINT does not have an implicit leading bit in its mantissa (unlike IEEE floats). In IEEE floating‐point, normalized numbers always have a leading 1, but in MXINT there may or may not be such a “leading 1.” The code checks the highest mantissa bit and, if it is not set, adjusts the exponent “by one” to mimic what would happen if there were a leading bit.

More in details, `dont_need_abs` is derived by checking the 6th bit (0x40) of the mantissa:
- if mantissa_abs & 0x40 is nonzero, the mantissa’s highest bit is set, so the value is already “normalized enough,” and no exponent shift is needed
- if it’s zero, the mantissa is “too small,” so the code effectively reduces the exponent by subtracting bias. bias is simply the same exponent and sign but with a zero fraction. Subtracting it is equivalent to dropping the exponent by 1 in bfloat16 terms.

`y[i] = dont_need_abs ? out : out - bias` => if the mantissa had its top bit set, we use "out" directly. Otherwise (the fraction was “too small”), we do "out - bias", effectively reducing the exponent by 1 in bfloat16 terms. This trick compensates for the missing “leading 1” bit and aligns the mantissa correctly for bfloat16 representation. Without this fix‐up step, you would not correctly reconstruct the intended magnitude for mantissas whose top bit is not set.


**c)** How does **`cta_tiler`** partition data for copying to shared memory in the **CUDA kernel**?  

**Answer:**
Let's start from the following call:

`// Tensor gX = local_tile(mX, cta_tiler, cta_coord);`

where:

 -`mX` is a 2D tensor view of your data with shape (group_size,num_groups).

 -`cta_tiler` is a shape (e.g., (BLK_M,BLK_K)) that defines how big each tile should be.

 -`cta_coord` is the 2D coordinate (blockIdx.x,blockIdx.y) identifying which tile this particular CTA (thread block) will handle.

The first step is transforming 1D data into a 2D tensor, in the line below:

`Tensor mX = flatten(flat_divide(mX_raw, group_tiler)); // mX has shape: (group_size, num_groups)`


where mX is viewed as 2D: the first dimension ranges over group_siz, and the second over num_groups=M/group_size.

Next step is defining the Tile Shape (cta_tiler)

The kernel then picks a tile shape: 

`auto BLK_M = Int<...>{}; auto BLK_K = Int<...>{}; auto cta_tiler = make_shape(BLK_M, BLK_K); `

After that, `local_tile(mX, cta_tiler, cta_coord)` computes the sub-tensor in mX corresponding to the tile of shape (BLK_M, BLK_K).

The `(blockIdx.x, blockIdx.y)` coordinate tells CUTE which tile along each dimension to select.

To sum up, `cta_tiler` (say (BLK_M, BLK_K)) describes the shape of the sub-matrix (tile) each CTA handles, then `local_tile(mX, cta_tiler, cta_coord)` extracts that sub-matrix (for the current blockIdx.x, blockIdx.y) from global memory and then the CTA copies this tile to shared memory (sX) for further processing.


**d)** How does **`layout_sX`** partition threads in a **thread block** for computation? _(Challenge)_  

**Answer:**
`layout_sX` is an indexing function that converts a 2D coordinate (m,k) to a 1D offset in shared memory.

How does the partition work:

 `Tensor tXsX = local_partition(sX, layout_sX, threadIdx.x); // (thd_m, thd_k)`

 -`sX`: A 2D tensor in shared memory with shape (BLK_M,BLK_K).

 -`layout_sX`: Describes how to map (m,k) coordinates onto a linear range [0,BLK_M ∗BLK_K).

 -`threadIdx.x`: The current thread’s “coordinate” in that linear range.


The local partition “splits” the region BLK_M, BLK_K so that each thread in the block is responsible for some subset of elements, so that each thread either handles one element (if blockDim.x = BLK_M * BLK_K), or handles a small sub-tile (multiple elements), depending on the tile shape vs. the number of threads.


**e)** Why is the saved **GPU memory** not exactly **(32 - (4+8/32))/32 = 86.7%** of the FP32 model?  

**Answer:**

The theoretical calculation assumes that all of the memory used for the linear layers is for the weight values, and that these weights can be reduced from 32 bits to 8 bits with only a small per-group overhead (4 bytes per group + an extra 8 bits per 32 weights).

This does not reflect the reality: not all of the GPU memory usage comes from storing weight tensors, so the measured savings percentage is lower than the theoretical ratio for MXINT8 weight compression. In theory, converting FP32 to MXINT4 with a single 8-bit exponent per 32 elements could achieve up to about 86.7% reduction of just the weight tensor storage. However the GPU memory usage also includes unquantized parameters (biases, embedding layers, LayerNorm weights, and the final classification layer remain in FP32), runtime memory (in inference, model.eval, some temporary tensors (activations, attention masks) still occupy memory... so when measuring peak memory usage,these intermediate allocations also contribute to the total).




The above calculation assumes converting FP32 weights to MXINT4 with a small per-group overhead (4 bytes per group plus an extra 8 bits per 32 weights). However, in the code, MXINT8 is being used, where weights are compressed from 32 bits to 8 bits. This gives a theoretical weight storage reduction of about 75%.

In practice, the overall saved GPU memory is lower than these ideal numbers because:

- partial compression: as shown in the code bleow, only the weights in the linear layers are quantized. Other parameters (such as biases, activation layers, LayerNorm weights and the final classification layer) remain in FP32.

```python
for layer_name, layer in model.named_modules():
    if not isinstance(layer, torch.nn.Linear):
        continue
    if "classifier" in layer_name:
        continue
    layer.cuda()
    layer_q = QLinearPacked.build_from_linear(layer, group_size=mxint8_group_size)
    set_layer_by_name(model, layer_name, layer_q)
    del layer
    torch.cuda.empty_cache()
```

- runtime memory: inference still allocates additional GPU memory for activations, intermediate buffers, and temporary tensors (like attention masks), which are not quantized.

- quantization overhead: even the small extra storage required for quantization metadata (like per-group scales or exponents) slightly offsets the savings on the weight tensors.

So, while the equation for MXINT4 predicts up to an 86.7% reduction in weight storage, using MXINT8 only achieves an ideal 75% reduction for the weights. When you also account for uncompressed parameters and runtime allocations, the overall GPU memory savings are significantly lower.