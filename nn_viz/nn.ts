export class NN {
    feature_set: string = "";
    num_features: number = 0;
    layers: Layer[] = [];

    constructor(buf: ArrayBufferLike) {
        const reader = new Reader(buf);

        this.feature_set = reader.readString();

        // prettier-ignore
        switch (this.feature_set) {
            case "half-compact": this.num_features = 192; break;
            case "half-piece": this.num_features = 768; break;
            case "half-king-piece": this.num_features = 40960; break;
        }

        const FT = 256;
        const L1 = 32;
        const L2 = 32;

        this.layers.push(Layer.read(reader, this.num_features, FT, 16, 16));
        this.layers.push(Layer.read(reader, 2 * FT, L1, 8, 32));
        this.layers.push(Layer.read(reader, L1, L2, 8, 32));
        this.layers.push(Layer.read(reader, L2, 1, 8, 32));

        console.assert(reader.isEOF(), "Failed to read entire NN file");
    }

    get ft() {
        return this.layers[0];
    }

    static async load(url: string) {
        let req = await fetch(url);
        let buf = await req.arrayBuffer();
        return new NN(buf);
    }
}

export class Layer {
    num_inputs!: number;
    num_outputs!: number;
    weight!: ArrayLike<number>;
    bias!: ArrayLike<number>;

    static read(
        reader: Reader,
        num_inputs: number,
        num_outputs: number,
        weight_bits: 8 | 16 | 32,
        bias_bits: 8 | 16 | 32
    ) {
        let layer = new Layer();

        layer.num_inputs = num_inputs;
        layer.num_outputs = num_outputs;
        layer.weight = reader.readIntArray(num_inputs * num_outputs, weight_bits);
        layer.bias = reader.readIntArray(num_outputs, bias_bits);

        return layer;
    }

    getWeightRow(index: number) {
        // weight in FT is stored as column-major
        let row = new Array(this.num_inputs);
        for (let i = 0; i < this.num_inputs; i++) {
            row[i] = this.weight[i * this.num_outputs + index];
        }
        return row;
    }

    getWeightFT(feature: number, neuron: number) {
        // weight in FT is stored as column-major
        return this.weight[feature * this.num_outputs + neuron];
    }
}

class Reader {
    private dv: DataView;
    private offset: number = 0;

    constructor(bin: ArrayBufferLike) {
        this.dv = new DataView(bin);
    }

    // Reads a 0 terminated string from a binary array
    readString() {
        let str = "";
        while (this.dv.getUint8(this.offset) !== 0) {
            str += String.fromCharCode(this.dv.getUint8(this.offset++));
        }
        this.offset++; // consume null byte
        return str;
    }

    readInt8Array(len: number) {
        let arr = new Int8Array(len);
        for (let i = 0; i < len; i++) {
            arr[i] = this.dv.getInt8(this.offset++);
        }
        return arr;
    }

    readInt16Array(len: number) {
        let arr = new Int16Array(len);
        for (let i = 0; i < len; i++) {
            arr[i] = this.dv.getInt16(this.offset, true);
            this.offset += 2;
        }
        return arr;
    }

    readInt32Array(len: number) {
        let arr = new Int32Array(len);
        for (let i = 0; i < len; i++) {
            arr[i] = this.dv.getInt32(this.offset, true);
            this.offset += 4;
        }
        return arr;
    }

    readIntArray(len: number, bits: 8 | 16 | 32) {
        // prettier-ignore
        switch (bits) {
            case 8: return this.readInt8Array(len);
            case 16: return this.readInt16Array(len);
            case 32: return this.readInt32Array(len);
        }
    }

    isEOF() {
        return this.offset === this.dv.byteLength;
    }
}
