module instantiate_ram ( 
	input  logic 		reset,
	input 				clk,

	output logic [10:0]  addr,
	output logic [3:0]	 wren,
	output logic [31:0]  data
);
							
	 

	logic [10:0] address;
	logic init_mem;

	always_ff @(posedge clk) begin
		if (reset) begin
			init_mem <= 1'b1;
			address <= '0;
		end else begin

			if (init_mem) begin
			     
			     if (address == (11'd600 - 1)) begin
					init_mem <= 1'b0;
				end
				
				else begin
				    address <= address + 11'd1;
				end
				
			end

		end
	end
		
	assign wren = {4{init_mem}};
	assign data = 32'h0000000000;
	assign addr = address;		

endmodule