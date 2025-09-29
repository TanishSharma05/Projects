module patch_extractor #(
    parameter int IMG_W = 28,
    parameter int K     = 3
)(
    input  logic clk,
    input  logic reset,
    input  logic start,                  // 1-cycle pulse
    input  logic [$clog2(IMG_W*IMG_W)-1:0] base_addr, // row*IMG_W + col
    input  logic [7:0]  memory_data,
    output logic [$clog2(IMG_W*IMG_W)-1:0] memory_addr,
    output logic        valid_out,       // unused by top; kept for completeness
    output logic signed [7:0] patch [0:K*K-1]
);
    typedef enum logic [3:0] {
        IDLE, R0, R1, R2, R3, R4, R5, R6, R7, R8, DONE, HOLD
    } st_t;
    st_t st, nx;
    logic [3:0] idx;
    logic signed [7:0] tmp [0:8];

    // Address offsets for 3x3 with stride=IMG_W
    logic [$clog2(IMG_W*IMG_W)-1:0] offs [0:8];
    always_comb begin
        offs[0]=0;          offs[1]=1;           offs[2]=2;
        offs[3]=IMG_W;      offs[4]=IMG_W+1;     offs[5]=IMG_W+2;
        offs[6]=2*IMG_W;    offs[7]=2*IMG_W+1;   offs[8]=2*IMG_W+2;
    end
    assign memory_addr = base_addr + offs[idx];

    // FSM
    always_ff @(posedge clk or posedge reset) begin
        if (reset) st <= IDLE;
        else       st <= nx;
    end
    always_comb begin
        nx = st;
        unique case (st)
            IDLE: nx = start ? R0 : IDLE;
            R0:R1; R1:R2; R2:R3; R3:R4; R4:R5; R5:R6; R6:R7; R7:R8; R8:DONE;
            DONE:HOLD;
            HOLD: nx = start ? HOLD : IDLE;
            default: nx = IDLE;
        endcase
    end

    // Capture data
    logic vld;
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            idx  <= 0;
            vld  <= 0;
        end else begin
            vld <= 0;
            if (st>=R0 && st<=R8) begin
                if (st!=R0) tmp[idx-1] <= memory_data;
                idx <= idx + 1;
            end
            if (st==DONE) begin
                tmp[8] <= memory_data;
                vld    <= 1;
                idx    <= 0;
            end
        end
    end
    assign valid_out = vld;
    assign patch = tmp;
endmodule
