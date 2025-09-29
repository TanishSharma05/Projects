module controller_fsm #(
    parameter int IMG_W = 28,
    parameter int IMG_H = 28,
    parameter int K     = 3
)(
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic patch_valid,  // ReLU valid indicates patch finished

    output logic [$clog2(IMG_H-K+1)-1:0] conv_row,
    output logic [$clog2(IMG_W-K+1)-1:0] conv_col,

    output logic patch_start,  // 1-cycle pulse to capture a patch
    output logic begin_pool,   // pulse to start pooling sweep
    output logic begin_flatten,// pulse to start flatten
    output logic begin_fc      // pulse to start FC compute
);
    localparam int CONV_W = IMG_W-K+1;
    localparam int CONV_H = IMG_H-K+1;

    typedef enum logic [2:0] {
        IDLE, CONV_SCAN, POOLING, FLATTEN, FC, DONE
    } state_t;

    state_t state, next;

    // Row/col iterators
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            conv_row <= '0;
            conv_col <= '0;
        end else if (state==CONV_SCAN && patch_valid) begin
            if (conv_col == CONV_W-1) begin
                conv_col <= '0;
                if (conv_row == CONV_H-1)
                    conv_row <= conv_row; // hold; transition will happen
                else
                    conv_row <= conv_row + 1;
            end else begin
                conv_col <= conv_col + 1;
            end
        end
    end

    // State reg
    always_ff @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next;
    end

    // Outputs & transitions
    always_comb begin
        patch_start   = 1'b0;
        begin_pool    = 1'b0;
        begin_flatten = 1'b0;
        begin_fc      = 1'b0;
        next          = state;

        unique case (state)
            IDLE: begin
                if (start) begin
                    patch_start = 1'b1; // first patch
                    next = CONV_SCAN;
                end
            end

            CONV_SCAN: begin
                // Emit a start pulse for each patch right after previous is accepted
                if (patch_valid) begin
                    if (conv_row == CONV_H-1 && conv_col == CONV_W-1) begin
                        next       = POOLING;
                    end else begin
                        patch_start = 1'b1;
                    end
                end
            end

            POOLING: begin
                begin_pool = 1'b1;
                next = FLATTEN;
            end

            FLATTEN: begin
                begin_flatten = 1'b1;
                next = FC;
            end

            FC: begin
                begin_fc = 1'b1;
                next = DONE;
            end

            DONE: begin
                next = IDLE;
            end
        endcase
    end
endmodule
