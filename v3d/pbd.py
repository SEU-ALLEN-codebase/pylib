import os
import struct
import numpy as np
import sys


class PBD:
    """
    by Zuohan Zhao
    from neuron_annotator/utility/ImageLoaderBasic.cpp
    2022/6/23
    """
    def __init__(self):
        self.compression_pos = 0
        self.decompression_pos = 0
        self.decompression_prior = 0

    def decompress_pbd8(self, src: str, tgt: np.array, length: int):
        cp = 0
        dp = 0
        pva = 0
        pvb = 0
        while cp < length:
            value = struct.unpack('B', src[cp])[0]
            if value < 33:
                count = value + 1
                tgt[dp:dp + count] = struct.unpack('B' * count, src[cp + 1:cp + 1 + count])
                cp += count + 1
                dp += count
                decompression_prior = tgt[dp - 1]
            elif value < 128:
                left_to_fill = value - 32
                while left_to_fill > 0:
                    fill_num = min(left_to_fill, 4)
                    cp += 1
                    src_char = src[cp]
                    to_fill = tgt[dp]
            else:
                pass

    def update_compression_buffer8(self):
        look_ahead = self.compression_pos
        while look_ahead < len(self.compression_buffer):
            lav = struct.unpack('B', compression_buffer[look_ahead])[0]
            if lav < 33:
                if look_ahead + lav + 1 < len(compression_buffer):
                    look_ahead += lav + 2
                else:
                    break
            elif lav < 128:
                compressed_diff_entries = (lav - 33) / 4 + 1
                if look_ahead + compressed_diff_entries < len(self.compression_buffer):
                    look_ahead += 2
                else:
                    break
        compression_len = look_ahead - compression_pos
        if decompression_pos == 0:
            decompression_pos = decompression
        compression_pos = look_ahead
        decompression_pos +=

    def update_compression_buffer16():
        pass

    def update_compression_buffer3():
        pass

    def load_image(self, path: str):
        assert os.path.exists(path)
        self.decompression_prior = 0
        formatkey = "v3d_volume_pkbitdf_encod"
        with open(path, "rb") as f:
            filesize = os.path.getsize(path)
            assert (filesize >= len(formatkey) + 2 + 4 * 4 + 1)
            format = f.read(len(formatkey)).decode('utf-8')
            assert (format == formatkey)
            endianCodeData = f.read(1).decode('utf-8')
            if endianCodeData == 'B':
                endian = '>'
            elif endianCodeData == 'L':
                endian = '<'
            else:
                raise Exception('endian be big/little')
            datatype = struct.unpack(endian + 'h', f.read(2))[0]
            if datatype == 1 or datatype == 33:
                dt = 'u1'
            elif datatype == 2:
                dt = 'u2'
            else:
                raise Exception('datatype be 1/2/4')
            sz = struct.unpack(endian + 'iiii', f.read(4 * 4))
            tot = sz[0] * sz[1] * sz[2] * sz[3]
            header_sz = 4 * 4 + 2 + 1 + len(formatkey)
            self.remaining_bytes = compressed_bytes = filesize - header_sz
            self.max_decompression_size = tot if datatype == 33 else tot * datatype
            self.channel_len = sz[0] * sz[1] * sz[2]
            read_step_size_bytes = 1024 * 20000
            total_read_bytes = 0
            img = np.zeros(tot, dtype=dt)
            compression_buffer = b""
            compression_pos = 0
            decompression_pos = 0

            while remaining_bytes > 0:
                current_read_bytes = min(remaining_bytes, read_step_size_bytes)
                pbd3_current_channel = total_read_bytes / channel_len
                bytes2channel_bound = (pbd3_current_channel + 1) * channel_len - total_read_bytes
                current_read_bytes = min(current_read_bytes, bytes2channel_bound)
                compression_buffer += f.read(current_read_bytes)
                total_read_bytes += current_read_bytes
                remaining_bytes -= current_read_bytes
                if datatype == 1:
                    update
                elif datatype == 33:
                    update
                elif datatype == 2:
                    update
            return img.reshape(sz[-1:-5:-1])