export interface PoseSetJsonItem {
    t: number;
    d: number;
    p?: number[][];
    l?: number[][];
    r?: number[][];
    v: number[][];
    h?: (number[] | null)[];
    e?: {
        [key: string]: any;
    };
    mt?: number;
    md?: number;
}
