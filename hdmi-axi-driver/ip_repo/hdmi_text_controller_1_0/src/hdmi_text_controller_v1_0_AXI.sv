`timescale 1ns / 1ps
module hdmi_text_controller_v1_0_AXI #
(

    // Parameters of Axi Slave Bus Interface S_AXI

    // Width of S_AXI data bus
    parameter integer C_S_AXI_DATA_WIDTH	= 32,
    // Width of S_AXI address bus
    parameter integer C_S_AXI_ADDR_WIDTH	= 14
)
(
  
    // Global Clock Signal
    input logic  S_AXI_ACLK,
    // Global Reset Signal. This Signal is Active LOW
    input logic  S_AXI_ARESETN,
    // Write address (issued by master, acceped by Slave)
    input logic [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
    // Write channel Protection type. This signal indicates the
        // privilege and security level of the transaction, and whether
        // the transaction is a data access or an instruction access.
    input logic [2 : 0] S_AXI_AWPROT,
    // Write address valid. This signal indicates that the master signaling
        // valid write address and control information.
    input logic  S_AXI_AWVALID,
    // Write address ready. This signal indicates that the slave is ready
        // to accept an address and associated control signals.
    output logic  S_AXI_AWREADY,
    // Write data (issued by master, acceped by Slave) 
    input logic [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
    // Write strobes. This signal indicates which byte lanes hold
        // valid data. There is one write strobe bit for each eight
        // bits of the write data bus.    
    input logic [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
    // Write valid. This signal indicates that valid write
        // data and strobes are available.
    input logic  S_AXI_WVALID,
    // Write ready. This signal indicates that the slave
        // can accept the write data.
    output logic  S_AXI_WREADY,
    // Write response. This signal indicates the status
        // of the write transaction.
    output logic [1 : 0] S_AXI_BRESP,
    // Write response valid. This signal indicates that the channel
        // is signaling a valid write response.
    output logic  S_AXI_BVALID,
    // Response ready. This signal indicates that the master
        // can accept a write response.
    input logic  S_AXI_BREADY,
    // Read address (issued by master, acceped by Slave)
    input logic [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
    // Protection type. This signal indicates the privilege
        // and security level of the transaction, and whether the
        // transaction is a data access or an instruction access.
    input logic [2 : 0] S_AXI_ARPROT,
    // Read address valid. This signal indicates that the channel
        // is signaling valid read address and control information.
    input logic  S_AXI_ARVALID,
    // Read address ready. This signal indicates that the slave is
        // ready to accept an address and associated control signals.
    output logic  S_AXI_ARREADY,
    // Read data (issued by slave)
    output logic [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
    // Read response. This signal indicates the status of the
        // read transfer.
    output logic [1 : 0] S_AXI_RRESP,
    // Read valid. This signal indicates that the channel is
        // signaling the required read data.
    output logic  S_AXI_RVALID,
    // Read ready. This signal indicates that the master can
        // accept the read data and response information.
    input logic  S_AXI_RREADY,
    
    input logic  [9:0] DrawX,
    
    input logic  [9:0] DrawY,
    
    output logic is_inverted,
    
    output logic [10:0] font_rom_address,
    
    output logic [31:0] palette [8],
    
    output logic [31:0] ctrl_reg,
    
    output logic [10:0] write_addr,
    
    output logic [31:0] write_data,
    
    output logic [3:0] write_ena,
    
    output logic [10:0] read_addr,
    
    input logic [31:0] read_data
);

// AXI4LITE signals
logic  [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
logic  axi_awready;
logic  axi_wready;
logic  [1 : 0] 	axi_bresp;
logic  axi_bvalid;
logic  [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
logic  axi_arready;
logic  [C_S_AXI_DATA_WIDTH-1 : 0] 	axi_rdata;
logic  [1 : 0] 	axi_rresp;
logic  	axi_rvalid;

logic [31:0] data_temp;

// Example-specific design signals
// local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
// ADDR_LSB is used for addressing 32/64 bit registers/memories
// ADDR_LSB = 2 for 32 bits (n downto 2)
// ADDR_LSB = 3 for 64 bits (n downto 3)
localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
localparam integer OPT_MEM_ADDR_BITS = 9; // CHANGE THIS VALUE

logic	 slv_reg_rden;
logic	 slv_reg_wren;
logic [C_S_AXI_DATA_WIDTH-1:0]	 reg_data_out;
integer	 byte_index;
logic	 aw_en;

// I/O Connections assignments

assign S_AXI_AWREADY	= axi_awready;
assign S_AXI_WREADY	= axi_wready;
assign S_AXI_BRESP	= axi_bresp;
assign S_AXI_BVALID	= axi_bvalid;
assign S_AXI_ARREADY = axi_arready;
assign S_AXI_RDATA	= axi_rdata;
assign S_AXI_RRESP	= axi_rresp;
assign S_AXI_RVALID	= axi_rvalid;
// Implement axi_awready generation
// axi_awready is asserted for one S_AXI_ACLK clock cycle when both
// S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_awready is
// de-asserted when reset is low.

always_ff @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_awready <= 1'b0;
      aw_en <= 1'b1;
    end 
  else
    begin    
      if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
        begin
           
          axi_awready <= 1'b1;
          aw_en <= 1'b0;
        end
        else if (S_AXI_BREADY && axi_bvalid)
            begin
              aw_en <= 1'b1;
              axi_awready <= 1'b0;
            end
      else           
        begin
          axi_awready <= 1'b0;
        end
    end 
end       


always_ff @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_awaddr <= 0;
    end 
  else
    begin    
      if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
        begin
          // Write Address latching 
          axi_awaddr <= S_AXI_AWADDR;
        end
    end 
end       



always_ff @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_wready <= 1'b0;
    end 
  else
    begin    
      if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en )
        begin
         
          axi_wready <= 1'b1;
        end
      else
        begin
          axi_wready <= 1'b0;
        end
    end 
end       


assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

always_ff @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
        for (integer i = 0; i < 8; i++)
        begin
           palette[i] <= 0;
        end
        
//        read_addr <= 11'b00000000000;
        write_addr <= 11'b00000000000;
        write_ena <= 4'b0000;
    end
  else begin
  
    write_ena <= 4'b0000;
    
    if (slv_reg_wren) begin
 
      if(axi_awaddr[13] == 0) begin
      
        write_addr <= axi_awaddr[13:2];
        write_data <= read_data;
        
        for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
          if ( S_AXI_WSTRB[byte_index] == 1 ) begin

            write_data[byte_index*8 +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
            write_ena[byte_index] <= 1;
          end         
        end
      
      else begin
      
        for ( byte_index = 0; byte_index <= (C_S_AXI_DATA_WIDTH/8)-1; byte_index = byte_index+1 )
          if ( S_AXI_WSTRB[byte_index] == 1 ) begin

            palette[axi_awaddr[4:2]][byte_index*8 +: 8] <= S_AXI_WDATA[(byte_index*8) +: 8];
          end
      end
    end
  end
end    



always_ff @( posedge S_AXI_ACLK )
begin
  if ( S_AXI_ARESETN == 1'b0 )
    begin
      axi_bvalid  <= 0;
      axi_bresp   <= 2'b0;
    end 
  else
    begin    
      if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID)
        begin
          // indicates a valid write response is available
          axi_bvalid <= 1'b1;
          axi_bresp  <= 2'b0; 
        end                   
      else
        begin
          if (S_AXI_BREADY && axi_bvalid)   
            begin
              axi_bvalid <= 1'b0; 
            end  
        end
    end
end     

assign slv_reg_rden = axi_arready & S_AXI_ARVALID & ~axi_rvalid;
always_comb
begin
      // Address decoding for reading registers
     reg_data_out = palette[axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB]];
end

enum logic [2:0] {
		start,
		wait_1,
		wait_2,
		wait_3,
		read,
		read_response 
		
	} state, state_nxt;   // Internal state logic


	always_ff @ (posedge S_AXI_ACLK)
	begin
		if (~S_AXI_ARESETN) begin
			state <= start;
			axi_arready <= 0;
			axi_rvalid <= 0 ;
			axi_araddr <= 0;
			axi_rdata <= 0;
			read_addr <= 0;
			data_temp <= 0;
	   end
		else begin 
		
		// Assign relevant control signals based on current state
		case (state)
			start: begin
                if (S_AXI_ARVALID && ~axi_rvalid) begin
                
                    axi_araddr <= S_AXI_ARADDR;
                    axi_arready <= 1'b1;
                    
                    if(!S_AXI_ARADDR[13]) begin
                    
                        read_addr <= S_AXI_ARADDR[13:2];  // BRAM word-aligned address
                        state <= wait_1;
                    end
                    else
                        state <= read ;                                       
                end
            end 
			
			wait_1 : 
				begin 
				     axi_arready <= 1'b0;
				     state <= wait_2;
				end
			wait_2 : 
				begin
					state <= wait_3;
				end
			wait_3 : 
				begin 
					data_temp <= read_data;
					state <= read;
				end
				
			 read: begin
			     axi_arready <= 1'b0;
			     
			     if(!S_AXI_ARADDR[13])
                    axi_rdata <= data_temp;  // Capture BRAM output after wait
                    
                    
                 else if(S_AXI_ARADDR[13])
                    axi_rdata <= palette[axi_araddr[4:2]];
                    
                 else 
                    axi_rdata <= 0;
                    
                 axi_rvalid <= 1;
                 state <= read_response;
            end
				
			
			read_response : 
				begin 
				
				    if(axi_rvalid && S_AXI_RREADY) begin
				        
				        axi_rvalid <= 0;
				        state <= start;

				    end
				    
					
				end
				
			
 
			default : state <= start;
		endcase
	end 
	
	end


endmodule

