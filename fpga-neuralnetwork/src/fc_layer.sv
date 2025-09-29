module fc_layer (
    input  logic clk,
    input  logic reset,
    input  logic signed [15:0] input_vec [0:3],   // Flattened pooled output (4 elements)
    input  logic               valid_in,

    output logic signed [15:0] output_vec [0:1],  // FC output (2 neurons)
    output logic               valid_out
);

    // FC weight and bias ROMs (pre-trained, hardcoded or loaded separately)
    logic signed [7:0] weights [0:1][0:3];  // 2 neurons × 4 inputs each
    logic signed [15:0] biases  [0:1];

    // One-time initialization
    initial begin
        // Neuron 0 weights
        weights[0][0] = 8'sd2; weights[0][1] = -8'sd1;
        weights[0][2] = 8'sd3; weights[0][3] = 8'sd1;
        biases[0]     = 16'sd0;

        // Neuron 1 weights
        weights[1][0] = -8'sd2; weights[1][1] = 8'sd2;
        weights[1][2] = 8'sd1;  weights[1][3] = -8'sd3;
        biases[1]     = 16'sd5;
    end

    // Multiply and accumulate logic
    logic signed [15:0] result0, result1;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            valid_out <= 0;
        end else if (valid_in) begin
            // Neuron 0
            result0 = (input_vec[0] * weights[0][0]) +
                      (input_vec[1] * weights[0][1]) +
                      (input_vec[2] * weights[0][2]) +
                      (input_vec[3] * weights[0][3]) + biases[0];

            // Neuron 1
            result1 = (input_vec[0] * weights[1][0]) +
                      (input_vec[1] * weights[1][1]) +
                      (input_vec[2] * weights[1][2]) +
                      (input_vec[3] * weights[1][3]) + biases[1];

            // Output assignment
            output_vec[0] <= result0;
            output_vec[1] <= result1;
            valid_out     <= 1;
        end else begin
            valid_out <= 0;
        end
    end

endmodule
