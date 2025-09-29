module pooling_window_controller #(
    parameter int ACT_W  = 26,
    parameter int ACT_H  = 26,
    parameter int POOL_K = 2,
    parameter int POOL_S = 2
)(
    input  logic clk,
    input  logic reset,
    input  logic signed [15:0] act_map [0:ACT_W*ACT_H-1],
    input  logic start,

    output logic signed [15:0] out0, out1, out2, out3, // current 2x2 window
    output logic               valid_out,
    output logic               done
);
    localparam int OUT_W = (ACT_W - POOL_K)/POOL_S + 1;
    localparam int OUT_H = (ACT_H - POOL_K)/POOL_S + 1;

    typedef enum logic [1:0] {IDLE, LOAD, WAIT, NEXT} st_t;
    st_t st, nx;

    logic [$clog2(OUT_H)-1:0] row;
    logic [$clog2(OUT_W)-1:0] col;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) st <= IDLE;
        else       st <= nx;
    end

    always_comb begin
        nx = st; valid_out = 0;
        unique case (st)
            IDLE: nx = start ? LOAD : IDLE;
            LOAD: begin valid_out = 1; nx = WAIT; end
            WAIT: nx = NEXT;
            NEXT: nx = (row==OUT_H-1 && col==OUT_W-1) ? IDLE : LOAD;
        endcase
    end

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin row<='0; col<='0; end
        else if (st==NEXT) begin
            if (col < OUT_W-1) col <= col + 1;
            else begin col <= 0; row <= row + 1; end
        end else if (st==IDLE && start) begin
            row <= '0; col <= '0;
        end
    end

    // base index in act_map for top-left of 2x2 window
    logic [$clog2(ACT_W*ACT_H)-1:0] base;
    always_comb begin
        base = (row*POOL_S)*ACT_W + (col*POOL_S);
    end

    always_ff @(posedge clk) begin
        if (st==LOAD) begin
            out0 <= act_map[base];
            out1 <= act_map[base + 1];
            out2 <= act_map[base + ACT_W];
            out3 <= act_map[base + ACT_W + 1];
        end
    end

    assign done = (st==NEXT) && (row==OUT_H-1) && (col==OUT_W-1);
endmodule
