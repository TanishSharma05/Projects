module flattener #(
    parameter int N = 169
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic signed [15:0] pooled_map [0:N-1],
    output logic signed [15:0] fc_input,
    output logic               valid_out,
    output logic               done
);
    logic [$clog2(N):0] idx;
    logic active;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            idx <= '0; active <= 0; valid_out <= 0;
        end else begin
            if (start) begin
                active <= 1; idx <= '0; valid_out <= 1;
            end else if (active) begin
                if (idx == N-1) begin
                    active <= 0; valid_out <= 0;
                end else idx <= idx + 1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (active) fc_input <= pooled_map[idx];
    end

    assign done = active && (idx==N-1);
endmodule
