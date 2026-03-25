declare module "lucide-react" {
  import * as React from "react";

  type LucideProps = React.SVGProps<SVGSVGElement> & {
    size?: number | string;
  };

  export const Globe: React.FC<LucideProps>;
  export const Server: React.FC<LucideProps>;
  export const Database: React.FC<LucideProps>;
  export const Zap: React.FC<LucideProps>;
  export const Layers: React.FC<LucideProps>;
}
