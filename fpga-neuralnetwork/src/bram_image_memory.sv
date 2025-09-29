module bram_image_memory #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 4  // 2? = 16 ? good for 4x4 image
) (
    input  logic                  clk,
    
    // Write interface
    input  logic                  write_en,
    input  logic [ADDR_WIDTH-1:0] write_addr,
    input  logic [DATA_WIDTH-1:0] write_data,

    // Read interface
    input  logic [ADDR_WIDTH-1:0] read_addr,
    output logic [DATA_WIDTH-1:0] read_data
);

    logic [DATA_WIDTH-1:0] memory [0:(1 << ADDR_WIDTH)-1];

    always_ff @(posedge clk) begin
        if (write_en)
            memory[write_addr] <= write_data;

        read_data <= memory[read_addr];  // always give output on read_addr
    end

endmodule
