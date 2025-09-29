module fc_layer #(
    parameter int N_IN  = 169,
    parameter int N_OUT = 2
)(
    input  logic clk,
    input  logic reset,
    input  logic signed [15:0] input_vec [0:N_IN-1],
    input  logic               valid_in,
    output logic signed [15:0] output_vec [0:N_OUT-1],
    output logic               valid_out
);
    // 8-bit weights + 16-bit bias
    logic signed [7:0]  weights [0:N_OUT-1][0:N_IN-1];
    logic signed [15:0] biases  [0:N_OUT-1];

    // Simple deterministic init (not trained): neuron0 all +1, neuron1 alternating +1/-1
    initial begin
        for (int o=0;o<N_OUT;o++) begin
            for (int i=0;i<N_IN;i++) begin
                if (o==0) weights[o][i] = 8'sd1;
                else      weights[o][i] = (i[0]==1'b0) ? 8'sd1 : -8'sd1;
            end
        end
        for (int o=0;o<N_OUT;o++) biases[o] = 16'sd0;
    end

    logic signed [31:0] acc [0:N_OUT-1]; // wider to avoid overflow during MAC

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            valid_out <= 0;
        end else if (valid_in) begin
            for (int o=0;o<N_OUT;o++) begin
                acc[o] = 32'sd0;
                for (int i=0;i<N_IN;i++)
                    acc[o] += input_vec[i] * weights[o][i];
                acc[o] += biases[o];
                // Truncate/saturate to 16-bit (simple truncate here)
                output_vec[o] <= acc[o][15:0];
            end
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
endmodule
