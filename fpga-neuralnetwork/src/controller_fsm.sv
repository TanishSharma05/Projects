module controller_fsm (
    input  logic clk,
    input  logic reset,
    input  logic start,
    input  logic patch_valid,  // Handshake from patch_extractor

    output logic [3:0] pixel_addr,
    output logic [1:0] kernel_sel,
    output logic       load_conv,
    output logic       load_pool,
    output logic       load_fc
);

    typedef enum logic [3:0] {
        IDLE,
        LOAD_IMAGE_PATCH,
        CONVOLVE,
        ACTIVATE,
        POOL,
        FLATTEN,
        FC,
        OUTPUT,
        NEXT_PATCH,
        DONE
    } state_t;

    state_t state, next_state;

    logic [1:0] row, col;

    // Address of top-left pixel of patch
    assign pixel_addr = (row * 4) + col;
    assign kernel_sel = 2'd0;

    // FSM transitions
    always_ff @(posedge clk or posedge reset) begin
        if (reset)
            state <= IDLE;
        else
            state <= next_state;
    end

    // FSM control logic
    always_comb begin
        load_conv = 0;
        load_pool = 0;
        load_fc   = 0;
        next_state = state;

        case (state)
            IDLE:
                if (start) next_state = LOAD_IMAGE_PATCH;

            LOAD_IMAGE_PATCH:
                next_state = CONVOLVE;

            CONVOLVE: begin
                load_conv = 1;
                if (patch_valid)
                    next_state = ACTIVATE;
            end

            ACTIVATE:
                next_state = POOL;

            POOL: begin
                load_pool = 1;
                next_state = FLATTEN;
            end

            FLATTEN:
                next_state = FC;

            FC: begin
                load_fc = 1;
                next_state = OUTPUT;
            end

            OUTPUT:
                next_state = NEXT_PATCH;

            NEXT_PATCH:
                next_state = (row == 1 && col == 1) ? DONE : LOAD_IMAGE_PATCH;

            DONE:
                next_state = IDLE;

            default:
                next_state = IDLE;
        endcase
    end

    // Row/col patch sliding logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            row <= 0;
            col <= 0;
        end else if (state == NEXT_PATCH) begin
            // Only increment if not at final patch
            if (!(row == 1 && col == 1)) begin
                if (col == 1) begin
                    col <= 0;
                    row <= row + 1;
                end else begin
                    col <= col + 1;
                end
            end
        end
    end

endmodule
