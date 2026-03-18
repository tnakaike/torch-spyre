// Input:  [512, 1024] ==> [128, 1024] * 4
// Output: [1, 1024]   ==> [1, 1024] * 1 (reduced to core 0)
module {
  func.func @sum(%input_hbm_address : index, %output_hbm_address : index) {
    %c0 = arith.constant 0 : index

    %output_lx_address_0 = arith.constant 0 : index

    %tile_size = arith.constant 131072 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %input_hbm_memref = ktdp.construct_memory_view %input_hbm_address, sizes: [512, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (0 <= d0 < 512, 0 <= d1 < 1024)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<512x1024xf16>

    %output_hbm_memref = ktdp.construct_memory_view %output_hbm_address, sizes: [1, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (0 <= d0 < 1, 0 <= d1 < 1024)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<512x1024xf16>

    %input_tile = ktdp.construct_access_tile %input_hbm_memref[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (0 <= d0 < 128, 0 <= d1 < 1024)>
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<512x1024xf16> -> !ktdp.access_tile<128x1024xindex>

    %input_tensor = ktdp.load %input_tile : !ktdp.access_tile<128x1024xindex> -> tensor<128x1024xf16>

    // ----- Version 1: Introduce atomic operations ----- //
    // Inner-core reduction
    %cf0 = arith.constant 0.0 : f16
    %output_tensor = tensor.splat %cf0 : tensor<1x1024xf16>
    %sum_tensor = linalg.reduce
      ins(%input_tensor : tensor<128x1024xf16>)
      outs(%output_tensor : tensor<1x1024xf16>)
      dimensions = [0]
      (%in: f16, %out: f16) {
        %0 = arith.addf %out, %in: f16
        linalg.yield %0: f16
      }

    // Inter-core reduction
    %output_lx0_memref = ktdp.construct_memory_view %output_lx_address_0, sizes: [1, 1024], strides: [1024, 1] {
        coordinate_set = affine_set<(d0, d1) : (0 <= d0 < 1, 0 <= d1 < 1024)>,
        memory_space = #ktdp.spyre_memory_space<LX0>
    } : memref<1x1024xf16>
    %output_lx0_tile = ktdp.construct_access_tile %output_lx_memref[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (0 <= d0 < 1, 0 <= d1 < 1024)>
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<1x1024xf16> -> !ktdp.access_tile<1x1024xindex>

    // %output_lx0_tensor = ktdp.load %output_lx0_tile : !ktdp.access_tile<1x1024xindex> -> tensor<1x1024xf16>
    // %acc_tensor = arith.addf %output_lx0_tensor, %sum_tensor : tensor<1x1024xf16>
    // ktdp.store %acc_tensor, %output_lx0_tile : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

    ktdp.atomic_rmw "addf" %sum_tensor, %output_lx0_tile : tensor<1x1024xf16>, !ktdp.access_tile<1x1024xindex>

    // Ineffecient for spyre due to memory access conflicts to LX0.

    // ----- End of Version 1 ----- //

    // ----- Version 2: Introduce a transfer operation for core-to-core communication ----- //
    // Inner-core reduction
    %cf0 = arith.constant 0.0 : f16
    %output_tensor = tensor.splat %cf0 : tensor<1x1024xf16>
    %sum_tensor = linalg.reduce
      ins(%input_tensor : tensor<128x1024xf16>)
      outs(%output_tensor : tensor<1x1024xf16>)
      dimensions = [0]
      (%in: f16, %out: f16) {
        %0 = arith.addf %out, %in: f16
        linalg.yield %0: f16
      }

    // Inter-core reduction (tree)
    // ktdp.transfer [sender tile ids], [receiver tile ids]
    // If my tile id does not includes in the lists, any transfer does not happen.
    %sum_tensor_2 = ktdp.transfer %sum_tensor [%c1, %c3], [%c0, %c2] {
      ^receiver_block(%received_tensor : tensor<1x1024xf16>) {
        %sum_tensor_1 = arith.addf %sum_tensor, %received_tensor : tensor<1x1024xf16>
        ktdp.transfer_yield %sum_tensor_1
      }
      ^default_block(%arg : tensor<1x1024xf16>) {
        ktdp.transfer_yield %arg
      }
    }
    %sum_tensor_4 = ktdp.transfer %sum_tensor_2 [%c2], [%c0] {
      ^receiver_block(%received_tensor : tensor<1x1024xf16>) {
        %sum_tensor_3 = arith.addf %sum_tensor_2, %received_tensor : tensor<1x1024xf16>
        ktdp.transfer_yield %sum_tensor_3
      }
      ^default_block(%arg : tensor<1x1024xf16>) {
        ktdp.transfer_yield %arg
      }
    }
    %is_core_0 = arith.cmpi %id, %c0
    scf.if %is_core_0 {
      ktdp.store %sum_tensor_4, %output_hbm_tile : tensor<1x1024xf16>, tile<1x1024xindex>
    }
    // ----- End of Version 2 ----- //

    return
  }
}
