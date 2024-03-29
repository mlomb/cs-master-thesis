import { Layer } from "./nn";

const COLORS = <const>["w", "b"];
const ROLES = <const>["P", "N", "B", "R", "Q", "K"];

type Color = (typeof COLORS)[number];
type Role = (typeof ROLES)[number];

// Cache coordinates to be used by other functions without having to pass them around
const FEATURES_COORDS = new Map<string, [number, number]>();
// Piece images
const PIECES = new Map<string, HTMLImageElement>();

export async function drawWeights(
    ctx: CanvasRenderingContext2D,
    data: ArrayLike<number>,
    width: number,
    height: number
) {
    let imageData = ctx.createImageData(width, height);
    let pixels = imageData.data;

    for (let i = 0; i < data.length; i++) {
        if (data[i] < 0) {
            pixels[i * 4 + 0] = -data[i];
            pixels[i * 4 + 1] = 0;
        } else {
            pixels[i * 4 + 0] = 0;
            pixels[i * 4 + 1] = data[i];
        }
        pixels[i * 4 + 2] = 0;
        pixels[i * 4 + 3] = 255;
    }

    let imageBitmap = await createImageBitmap(imageData);

    ctx.drawImage(imageBitmap, 0, 0);
}

export function drawNeurons(
    ctx: CanvasRenderingContext2D,
    offset_x: number,
    offset_y: number,
    layer: Layer,
    neuron: number
) {
    // grid of sqrt(count) x sqrt(count) neurons
    let size = 4; // Math.ceil(Math.sqrt(count));
    let spacing = 15;
    let radius = 5;

    for (let i = 0; i < layer.num_outputs; i++) {
        let x = offset_x + (i % size) * spacing;
        let y = offset_y + Math.floor(i / size) * spacing;

        if (i == neuron) {
            ctx.lineWidth = 1;

            // connect from 0,0 to x,y 768 inputs
            for (let feature = 0; feature < 768; feature++) {
                if (!FEATURES_COORDS.has(feature + "")) continue;
                let [x1, y1] = FEATURES_COORDS.get(feature + "")!;

                let value = layer.getWeightFT(feature, neuron);
                let color = "rgba(" + (value > 0 ? "0, 255, 0" : "255, 0, 0") + ", " + Math.abs(value / 300) + ")";

                ctx.strokeStyle = color;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        }

        ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fill();
    }
}

type Cell = {
    // piece
    role?: Role;
    color?: Color;
    opacity?: number;

    // other
    text?: string;
    accent?: string;

    coords_key?: string;
};

const BOARD_CELL_SIZE = 16;

export function drawBoard(ctx: CanvasRenderingContext2D, offset_x: number, offset_y: number, cells: Cell[]) {
    console.assert(cells.length == 64, "Invalid number of cells");

    // draw chess board
    let color1 = "#f0d9b5";
    let color2 = "#b58863";

    for (let square = 0; square < 64; square++) {
        let col = square % 8;
        let row = Math.floor(square / 8);

        let x = col * BOARD_CELL_SIZE + offset_x;
        let y = row * BOARD_CELL_SIZE + offset_y;

        ctx.fillStyle = (col + row) % 2 == 0 ? color1 : color2;
        ctx.fillRect(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);

        let { role, color, opacity, text, accent, coords_key } = cells[square];

        if (accent) {
            ctx.fillStyle = accent;
            ctx.fillRect(x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
        }

        if (role && color) {
            let key = color + role;
            let img = PIECES.get(key);
            if (img) {
                ctx.globalAlpha = opacity === undefined ? 1 : opacity;
                ctx.drawImage(img, x, y, BOARD_CELL_SIZE, BOARD_CELL_SIZE);
                ctx.globalAlpha = 1;
            }
        }

        if (text) {
            ctx.textAlign = "left";
            ctx.font = "4px Arial";
            ctx.fillStyle = "white";
            ctx.fillText(`${text}`, x + 1, y + BOARD_CELL_SIZE - 1);
        }

        if (coords_key) {
            FEATURES_COORDS.set(coords_key, [x + BOARD_CELL_SIZE / 2, y + BOARD_CELL_SIZE / 2]);
        }
    }
}

export function drawHalfPiece(ctx: CanvasRenderingContext2D, layer: Layer, neuron: number) {
    for (let channel = 0; channel < 12; channel++) {
        drawBoard(
            ctx,
            0,
            channel * 150,
            Array.from({ length: 64 }, (_, i) => {
                let feature = 64 * channel + i;
                let value = layer.getWeightFT(feature, neuron);
                let opacity = Math.abs(value / 300);

                return {
                    text: `${feature}`,
                    accent: "rgba(" + (value > 0 ? "0, 255, 0" : "255, 0, 0") + ", " + opacity + ")",
                    coords_key: `${feature}`,

                    color: channel < 6 ? "w" : "b",
                    role: ROLES[channel % 6],
                    opacity,
                };
            })
        );
    }

    ctx.font = "20px monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = "white";
    ctx.fillText("Half-Piece", (BOARD_CELL_SIZE * 8) / 2, -30);
    ctx.fillText("[768]", (BOARD_CELL_SIZE * 8) / 2, -10);
}

async function loadPieces() {
    for (let color of COLORS) {
        for (let role of ROLES) {
            let img = new Image();
            img.src = `pieces/${color}${role}.svg`;
            PIECES.set(color + role, img);
        }
    }
}

loadPieces();
